from maple.navigation.drive_control import PIDController

# Test function
def test_pid_max_value():
    pid = PIDController(kp=2.0, ki=1.0, kd=0.5, setpoint=0, max_value=5)
    dt = 0.1

    print("---- Testing saturation ----")
    # Force large error to exceed max_value
    output = pid.update(measurement=100, dt=dt)
    print(f"Output with large error: {output}")
    assert abs(output) <= 5, "Output exceeds max_value!"
    assert pid.integral == 0, "Integral should be reset to prevent windup"

    print("---- Testing recovery ----")
    # Reduce error so output falls within bounds again
    output = pid.update(measurement=2, dt=dt)
    output = pid.update(measurement=2, dt=dt)
    print(f"Output after recovery: {output}")
    assert abs(output) < 5, "Output should be under max_value after recovery"
    assert pid.integral != 0, "Integral should resume accumulating after saturation ends"

    print("All tests passed.")