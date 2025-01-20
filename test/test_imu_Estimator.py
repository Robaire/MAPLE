# Do I have to import the imu_Estimator function? Let's find out!

from test.mock_agent import mock_agent
from pose.imu_Estimator import imu_Estimator
import pytransform3d.transformations as pytr
import pytransform3d.rotations as pyrot
import numpy as np

def test_imu_integration(mock_agent):
    """
    Test to see if the change_in_state_imu_frame output works. If this fails, double-check any assumptions about the imu orientation and data format.
    """
    # Create an imu_Estimator
    imu_estimator = imu_Estimator(mock_agent)
    dt = imu_estimator.dt

    # Create synthetic imu data
    imu_data_none = [0, 0, 0, 0, 0, 0]
    imu_data = [1/(dt*dt), 1/(dt*dt), 1/(dt*dt), np.pi/dt, np.pi/dt, np.pi/dt]

    # Test no state change
    mock_agent.get_imu_data.return_value = imu_data_none
    assert np.all(imu_estimator.change_in_state_imu_frame() == create_identity_pytransform())
    # Test a positive state change
    mock_agent.get_imu_data.return_value = imu_data
    expected_state = pytr.transform_from(pyrot.active_matrix_from_extrinsic_roll_pitch_yaw([np.pi, np.pi, np.pi]),[1,1,1])
    assert np.all(imu_estimator.change_in_state_imu_frame() == expected_state)

def test_imu_next_state(mock_agent):
    """
    Test to see if the next_state output works. In particular, check to see if the transforms are combined correctly.
    """
    # Create an imu_Estimator
    imu_estimator = imu_Estimator(mock_agent)

    # Create synthetic imu data
    imu_data = [1, 1, 1, 1, 1, 1]

    mock_agent.get_imu_data.return_value = imu_data
    mock_agent.prev_state = create_identity_pytransform()
    next_state = imu_estimator.next_state()
    expected_next_state = pytr.concat(mock_agent.prev_state, imu_estimator.change_in_state_imu_frame())
    assert np.all(next_state == expected_next_state)

def create_identity_pytransform():
    """
    Create an identity transform
    """
    return pytr.transform_from(pyrot.active_matrix_from_extrinsic_roll_pitch_yaw([0, 0, 0]), [0, 0, 0])
