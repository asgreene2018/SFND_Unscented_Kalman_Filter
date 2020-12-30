#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // Laser measurement covariance matrix
  R_laser_ = MatrixXd(2,2);
  R_laser_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  // Laser state transition matrix
  H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  // Radar measurement covariance matrix
  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;

  // Process noise covariance matrix
  Q_ = MatrixXd(2, 2);
  Q_ << std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;
  

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Used to track when first measurement is ingested
  is_initialized_ = false;

  // Predicted sigma point matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Initialize current timestamp
  time_us_ = 0.0;

  // Initialize weights for mean/covariance prediction
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for(int i = 1; i < 2*n_aug_+1; i++)
  {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  // First measurement logic
  if(!is_initialized_)
  {
    // If lidar data, use first two dimensions to initialize target position
    if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0,
            0,
            0;
    }

    // If radar data, use first three dimensions to initialize target radial distance, speed, and direction
    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      double x = rho*cos(phi);
      double y = rho*sin(phi);
      double vx = rho_dot*cos(phi);
      double vy = rho_dot*sin(phi);
      double v = sqrt(vx*vx + vy*vy);

      x_ << x, y, v, rho, rho_dot;
    }

    // Otherwise, alert user that no information has been accepted
    else
    {
      std::cout << "No Measurement recorded" << std::endl;
    }

    // Record timestamp
    time_us_ = meas_package.timestamp_;

    // Mark initialization complete
    is_initialized_ = true;
  }

  // All other measurements
  else
  {
    // Get time step
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;

    // Update time elapse from start
    time_us_ = meas_package.timestamp_;

    // Prediction step
    Prediction(dt);

    // Measurement update step
    if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
      UpdateLidar(meas_package);
    }

    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    {
      UpdateRadar(meas_package);
    }

  }
}

void UKF::GenerateSigmaPoints(MatrixXd& Xsig_out)
{
  // Instantiate augmented state vector
  VectorXd x_aug_ = VectorXd(n_aug_);
  x_aug_.fill(0.0);
  x_aug_.head(n_x_) = x_;

  // Instantiate augmented covariance matrix
  MatrixXd P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_.bottomRightCorner(2,2) = Q_;

  // Get square root of augmented covariance matrix thorugh Cholesky decomposition
  MatrixXd L = P_aug_.llt().matrixL();

  // Calculate sigma point matrix
  Xsig_out.col(0) = x_aug_;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_out.col(i+1)      = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_out.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_out, double delta_t, MatrixXd& Xsig_aug_)
{
  // Predict each new sigma point
  for(int i = 0; i < 2*n_aug_+1; i++)
  {
    // Unpack state elements from ith row
    double px = Xsig_aug_(0, i);
    double py = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6,i);

    // Predict mean 2D position
    double px_new, py_new;
    if(std::fabs(yawd) > 0.001)
    {
      px_new = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_new = py + v/yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    }
    else
    {
      px_new = px + v*delta_t*cos(yaw);
      py_new = py + v*delta_t*sin(yaw);
    }

    // Predict mean speed, yaw, and yaw rate
    double v_new = v;
    double yaw_new = yaw + yawd*delta_t;
    double yawd_new = yawd;

    // Add stochasticity portion to predictions
    px_new = px_new + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_new = py_new + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_new = v_new + nu_a * delta_t;
    yaw_new = yaw_new + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_new = yawd_new + nu_yawdd * delta_t;

    // Write new sigma points into ith column of sigma prediction matrix
    Xsig_out(0, i) = px_new;
    Xsig_out(1, i) = py_new;
    Xsig_out(2, i) = v_new;
    Xsig_out(3, i) = yaw_new;
    Xsig_out(4, i) = yawd_new;
  }
}

void UKF::PredictMeanAndCovariance()
{
  // Calculate new state matrix from weighted sigma points
  x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) 
  {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // Calculate new state covariance matrix
  P_.fill(0.0);
  for(int i = 0; i < 2*n_aug_ + 1; i++)
  {
    VectorXd xDiff = Xsig_pred_.col(i) - x_;
    while(xDiff(3) > M_PI)
      xDiff(3) -= 2.*M_PI;
    while(xDiff(3) < -M_PI)
      xDiff(3) += 2.*M_PI;
    
    P_ = P_ + weights_(i) * xDiff * xDiff.transpose();
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // Get augmented sigma point matrix
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2*n_aug_ + 1);
  GenerateSigmaPoints(Xsig_aug_);

  // Predict new sigma points
  SigmaPointPrediction(Xsig_pred_, delta_t, Xsig_aug_);

  // Predict mean and covariance
  PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // Set measurement dimension
  int n_z = 2;

  // Convert sigma point into laser measurements
  MatrixXd zSig = H_laser_*Xsig_pred_;

  // Calculate measurement mean
  VectorXd zPred = zSig * weights_;

  // Calcuate measurement covariance
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for(int i = 0; i < 2*n_aug_ + 1; i++)
  {
    VectorXd zDiff = zSig.col(i) - zPred;

    while(zDiff(1) > M_PI)
      zDiff(1) -= 2.0*M_PI;
    while(zDiff(1) < -M_PI)
      zDiff(1) += 2.0*M_PI;

    S += weights_(i) * zDiff * zDiff.transpose();

  }

  S = S + R_laser_;

  // Calculate cross correlation matrix
  MatrixXd T = MatrixXd::Zero(n_x_, n_z);
  for(int i = 0; i < 2*n_aug_+1; i++)
  {
    VectorXd zDiff = zSig.col(i) - zPred;
    while(zDiff(1) > M_PI)
      zDiff(1) -= 2.0*M_PI;
    while(zDiff(1) < -M_PI)
      zDiff(1) += 2.0*M_PI;
    
    VectorXd xDiff = Xsig_pred_.col(i) - x_;
    while(xDiff(3) > M_PI)
      xDiff(3) -= 2.0*M_PI;
    while(xDiff(3) < -M_PI)
      xDiff(3) += 2.0*M_PI;
    
    T = T + weights_(i) * xDiff * zDiff.transpose();
  }

  // Calculate Kalman gain
  MatrixXd K = T * S.inverse();

  // Calculate difference between predictions and measurements
  VectorXd zDiff = meas_package.raw_measurements_ - zPred;
  while(zDiff(1) > M_PI)
    zDiff(1) -= 2.0*M_PI;
  while(zDiff(1) < -M_PI)
    zDiff(1) += 2.0*M_PI;
  
  // Update state and covariance
  x_ = x_ + K * zDiff;
  P_ = P_ - K * S * K.transpose();

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // Set measurement dimension
  int n_z = 3;

  // Convert sigma point into radar measurements
  MatrixXd zSig = MatrixXd(n_z, 2*n_aug_+1);
  for(int i = 0; i < 2*n_aug_+1; i++)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    
    zSig(0, i) = sqrt(px*px + py*py);
    zSig(1, i) = atan2(py, px);
    zSig(2, i) = (px*cos(yaw)*v + py*sin(yaw)*v) / sqrt(px*px + py*py);
  }
  // Calculate measurement mean
  VectorXd zPred = zSig * weights_;

  // Calcuate measurement covariance
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for(int i = 0; i < 2*n_aug_ + 1; i++)
  {
    VectorXd zDiff = zSig.col(i) - zPred;

    while(zDiff(1) > M_PI)
      zDiff(1) -= 2.0*M_PI;
    while(zDiff(1) < -M_PI)
      zDiff(1) += 2.0*M_PI;

    S += weights_(i) * zDiff * zDiff.transpose();

  }

  S = S + R_radar_;

  // Calculate cross correlation matrix
  MatrixXd T = MatrixXd::Zero(n_x_, n_z);
  for(int i = 0; i < 2*n_aug_+1; i++)
  {
    VectorXd zDiff = zSig.col(i) - zPred;
    while(zDiff(1) > M_PI)
      zDiff(1) -= 2.0*M_PI;
    while(zDiff(1) < -M_PI)
      zDiff(1) += 2.0*M_PI;
    
    VectorXd xDiff = Xsig_pred_.col(i) - x_;
    while(xDiff(3) > M_PI)
      xDiff(3) -= 2.0*M_PI;
    while(xDiff(3) < -M_PI)
      xDiff(3) += 2.0*M_PI;

    T = T + weights_(i) * xDiff * zDiff.transpose();
  }

  // Calculate Kalman gain
  MatrixXd K = T * S.inverse();

  // Calculate difference between predictions and measurements
  VectorXd zDiff = meas_package.raw_measurements_ - zPred;
  while(zDiff(1) > M_PI)
    zDiff(1) -= 2.0*M_PI;
  while(zDiff(1) < -M_PI)
    zDiff(1) += 2.0*M_PI;
  
  // Update state and covariance
  x_ = x_ + K * zDiff;
  P_ = P_ - K * S * K.transpose();
}