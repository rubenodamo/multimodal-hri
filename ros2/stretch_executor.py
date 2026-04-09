"""
Stretch 3 ROS 2 command executor.

Subscribes to /hri/command, parses the JSON payload, and maps them to hardcoded Stretch manipulation primitives such as pick, place, and stop.

This node is designed for a constrained real-robot demo using fixed poses.

Run on the Stretch robot (with ROS 2 Humble sourced):
    python3 stretch_executor.py
"""

from __future__ import annotations

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

import stretch_body.robot as sb_robot


# Pose and timing constants
LIFT_TRAVEL = 0.95      # Safe travel height above the table (m)
LIFT_TABLE = 0.90       # Table surface height (m)
ARM_RETRACTED = 0.02    # Arm fully retracted (m)
ARM_REACH = 0.60        # Arm extension to reach object (m)
BASE_LEFT = 0.15        # Base rotation for left object position (rad)
BASE_RIGHT = -0.22      # Base rotation for right object position (rad)
GRIPPER_OPEN = 70.0     # Gripper open position (degrees)
GRIPPER_CLOSED =  0.0   # Gripper closed position (degrees)
SETTLE_LIFT = 2.5       # Time to settle after lift motion (s)
SETTLE_ARM = 2.0        # Time to settle after arm motion (s)
SETTLE_BASE = 2.0       # Time to settle after base motion (s)
SETTLE_GRIPPER = 1.0    # Time to settle after gripper motion (s)


class StretchExecutorNode(Node):
    """
    ROS 2 node that executes fixed robot primitives from incoming commands.
    
    Attributes:
        _robot: The stretch_body Robot instance for controlling the hardware.
        _motion_lock: A threading lock to ensure only one motion primitive runs at a time.
    """

    def __init__(self) -> None:
        """
        Initialise the StretchExecutorNode.
        """

        super().__init__("stretch_executor")

        self.get_logger().info("Starting up stretch_body robot…")
        self._robot = sb_robot.Robot()
        if not self._robot.startup():
            self.get_logger().error("stretch_body startup failed — check robot status")
            raise RuntimeError("stretch_body startup failed")
        self.get_logger().info("stretch_body ready — robot must already be homed")

        self._motion_lock = threading.Lock()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(String, "/hri/command", self._on_command, qos)
        self.get_logger().info(
            "Stretch executor ready — subscribed to /hri/command"
        )

    def _on_command(self, msg: String) -> None:
        """
        Callback for incoming command messages. Parses the JSON payload and dispatches to the appropriate primitive.
        
        Args:
            msg: The incoming ROS 2 String message containing the command as JSON.   
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid JSON: {msg.data!r}")
            return

        action     = payload.get("action", "unknown")
        obj        = payload.get("object") or "-"
        location   = payload.get("location") or "-"
        confidence = float(payload.get("confidence", 0.0))

        self.get_logger().info(
            f"Command received | action={action} object={obj} "
            f"location={location} confidence={confidence:.2f}"
        )

        DISPATCH = {
            "stop":        self._stop_all,
            "cancel":      self._stop_all,
            "pick":        self._pick_from_payload,
            "pick_left":   self._pick_left,
            "pick_right":  self._pick_right,
            "place":       self._place_from_payload,
            "place_left":  self._place_left,
            "place_right": self._place_right,
        }

        primitive = DISPATCH.get(action)
        if primitive:
            threading.Thread(
                target=self._run_primitive, args=(primitive, payload), daemon=True
            ).start()
        else:
            self.get_logger().warn(f"No primitive for action: {action!r}")

    def _run_primitive(self, fn, payload) -> None:
        """
        Execute a motion primitive if no other motion is active.
        
        Args:
            fn: The motion primitive function to execute.
            payload: The command payload to pass to the primitive function.
        """

        if not self._motion_lock.acquire(blocking=False):
            self.get_logger().warn("Motion already in progress — command ignored")
            return
        
        try:
            fn(payload)
        except Exception as exc:
            self.get_logger().error(f"Motion primitive failed: {exc}")
        finally:
            self._motion_lock.release()

    def _stop_all(self, _payload=None) -> None:
        """
        Stop all robot motion immediately.

        Args:
            _payload: Unused payload parameter.
        """

        self.get_logger().info("[stop] Stopping all motion")
        self._robot.base.stop()
        self._robot.lift.stop()
        self._robot.arm.stop()

    def _pick_from_payload(self, payload) -> None:
        """
        Route a generic 'pick' command to the appropriate primitive based on location.

        Args:
            payload: Command payload containing at least a 'location' field.
        """

        location = (payload.get("location") or "").lower()
        if location == "right":
            self._pick_right(payload)
        else:
            self._pick_left(payload) # default to left

    def _pick_left(self, _payload=None) -> None:
        """
        Execute a pick action at the fixed left position.

        Args:
            _payload: Unused payload parameter.
        """

        self.get_logger().info("[pick_left] Executing")

        # Prepare for approach
        self._set_gripper(GRIPPER_OPEN)
        self._move_lift(LIFT_TRAVEL)
        self._move_arm(ARM_RETRACTED)

        # Move to target
        self._rotate_base(BASE_LEFT)
        self._move_arm(ARM_REACH)
        self._move_lift(LIFT_TABLE)

        # Grasp and lift
        self._set_gripper(GRIPPER_CLOSED)
        self._move_lift(LIFT_TRAVEL)

        # Return to neutral
        self._move_arm(ARM_RETRACTED)
        self._rotate_base(-BASE_LEFT)

        self.get_logger().info("[pick_left] Done")

    def _pick_right(self, _payload=None) -> None:
        """
        Execute a pick action at the fixed right position.

        Args:
            _payload: Unused payload parameter.
        """

        self.get_logger().info("[pick_right] Executing")

        # Prepare for approach
        self._set_gripper(GRIPPER_OPEN)
        self._move_lift(LIFT_TRAVEL)
        self._move_arm(ARM_RETRACTED)

        # Move to target
        self._rotate_base(BASE_RIGHT)
        self._move_arm(ARM_REACH)
        self._move_lift(LIFT_TABLE)

        # Grasp and lift
        self._set_gripper(GRIPPER_CLOSED)
        self._move_lift(LIFT_TRAVEL)

        # Return to neutral
        self._move_arm(ARM_RETRACTED)
        self._rotate_base(-BASE_RIGHT)

        self.get_logger().info("[pick_right] Done")

    def _place_from_payload(self, payload) -> None:
        """
        Route a generic 'place' command to the appropriate primitive based on location.
        
        Args:
            payload: Command payload containing at least a 'location' field.
        """
        
        location = (payload.get("location") or "").lower()

        if location == "left":
            self._place_left(payload)
        else:
            self._place_right(payload)  # default to right

    def _place_left(self, _payload=None) -> None:
        """
        Execute a place action at the fixed left position.

        Args:
            _payload: Unused payload parameter.
        """

        self.get_logger().info("[place_left] Executing")

        self._move_lift(LIFT_TRAVEL)
        self._rotate_base(BASE_LEFT)

        self._move_arm(ARM_REACH)
        self._move_lift(LIFT_TABLE)
        self._set_gripper(GRIPPER_OPEN)

        self._move_lift(LIFT_TRAVEL)
        self._move_arm(ARM_RETRACTED)
        self._rotate_base(-BASE_LEFT)

        self.get_logger().info("[place_left] Done")

    def _place_right(self, _payload=None) -> None:
        """
        Execute a place action at the fixed right position.

        Args:
            _payload: Unused payload parameter.
        """
        
        self.get_logger().info("[place_right] Executing")

        # Move to positon
        self._move_lift(LIFT_TRAVEL)
        self._rotate_base(BASE_RIGHT)

        # Lower and release
        self._move_arm(ARM_REACH)
        self._move_lift(LIFT_TABLE)
        self._set_gripper(GRIPPER_OPEN)

        # Return to neutral
        self._move_lift(LIFT_TRAVEL)
        self._move_arm(ARM_RETRACTED)
        self._rotate_base(-BASE_RIGHT)

        self.get_logger().info("[place_right] Done")


    def _move_lift(self, pos_m: float) -> None:
        """
        Move the lift to a specified position.
        
        Args:
            pos_m: The target lift position in metres.
        """

        self._robot.lift.move_to(pos_m)
        self._robot.push_command()
        time.sleep(SETTLE_LIFT)

    def _move_arm(self, pos_m: float) -> None:
        """
        Move the arm to a specified positions.
        
        Args:
            pos_m: The target arm position in metres.
        """

        self._robot.arm.move_to(pos_m)
        self._robot.push_command()
        time.sleep(SETTLE_ARM)

    def _rotate_base(self, radians: float) -> None:
        """
        Rotate the base by a specified angle.

        Args:
            radians: Rotation angle in radians.
        """

        self._robot.base.rotate_by(radians)
        self._robot.push_command()
        time.sleep(SETTLE_BASE)

    def _set_gripper(self, pos: float) -> None:
        """
        Set the gripper opening position.
        Args:
            pos: Gripper position in degrees.
        """

        self._robot.end_of_arm.move_to("stretch_gripper", pos)
        self._robot.push_command()
        time.sleep(SETTLE_GRIPPER)


def main() -> None:
    """
    Initialise ROS 2 and run the StretchExecutorNode. 
    """

    rclpy.init()
    node = StretchExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._robot.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
