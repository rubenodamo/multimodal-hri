"""
Validation logic for RobotCommand objects.

This module checks that all required fields for a given action are present and valid.
"""

from models import ACTION_FIELD_REQUIREMENTS, Location, ObjectName, RobotCommand


def validate_command(cmd: RobotCommand) -> list[str]:
    """
    Validate required fields for a RobotCommand.

    Args:
        cmd: Command to validate.

    Returns:
        List of missing field names. Empty if valid.
        Returns ["action"] if no action is set.
    """

    if cmd.action is None:
        return ["action"]

    required_fields = ACTION_FIELD_REQUIREMENTS.get(cmd.action, [])
    errors: list[str] = []

    for field_name in required_fields:
        value = getattr(cmd, field_name, None)

        # Missing if None or explicitly set to "none"
        if value is None:
            errors.append(field_name)
        elif hasattr(value, "value") and value.value == "none":
            errors.append(field_name)

    return errors
