#include "plugin_undistort.h"
#include <Eigen/Dense>
#include <colors.h>
#include <image.h>

#include <dbg.h>

//! Makes accessing the arange matrix type easier
template <typename Scalar> struct arange_helper {
  using MatrixType =
      Eigen::Matrix<Scalar, Eigen::Dynamic,
                    /* columns */ 1, Eigen::ColMajor | Eigen::AutoAlign,
                    Eigen::Dynamic,
                    /*max column size */ 1>;
};

/** Generate a vector with all numbers in range [start, end).
 *
 *  Similar to numpy arange function. If end <= start then an empty
 *  vector will be generated.
 */
template <typename Scalar> class arange_dynamic_functor {
  const int start;

public:
  arange_dynamic_functor(const int start) : start(start) {}

  const Scalar operator()(Eigen::Index row) const {
    return Scalar(start + row);
  }
};

//! Applies arange_dynamic_functor to generate a vector of integers
template <typename Scalar = double>
Eigen::CwiseNullaryOp<arange_dynamic_functor<Scalar>,
                      typename arange_helper<Scalar>::MatrixType>
arange(const int start, const int end) {
  using MatrixType = typename arange_helper<Scalar>::MatrixType;
  const int size = std::max(end - start, 0);
  return MatrixType::NullaryExpr(size, 1,
                                 arange_dynamic_functor<Scalar>(start));
}

template <typename Scalar = double>
Eigen::Matrix<Scalar, Eigen::Dynamic, 2>
meshgrid(const int x_start, const int x_end, const int y_start,
         const int y_end) {
  using Eigen::Map;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;

  // assume the following parameters:
  //
  // x_start = 0
  // x_end = 3
  // y_start = -2
  // y_end = 0

  // given those params x_range should be
  //
  // [0, 1, 2].transpose()
  const VectorXd x_range = arange(x_start, x_end);
  // y_range should be
  // [-2, -1]
  const RowVectorXd y_range = arange(y_start, y_end).transpose();

  // x_mesh would be
  // [0, 1, 2, 0, 1, 2].tranpose()
  MatrixXd x_mesh = x_range.replicate(y_range.size(), 1);
  // y_duplicated would be
  // [[-2, -1],
  //  [-2, -1],
  //  [-2, -2]]
  MatrixXd y_duplicated = y_range.replicate(x_range.size(), 1);
  // y_mesh is
  // [-2, -2, -2, -1, -1, -1].tranpose()
  Map<VectorXd> y_mesh(y_duplicated.data(), y_duplicated.size());

  // xy_mesh is
  // [[0, -2]
  //  [1, -2],
  //  [2, -2],
  //  [0, -1]
  //  [1, -1],
  //  [2, -1]]
  Matrix<Scalar, Eigen::Dynamic, 2> xy_mesh(x_mesh.size(), 2);
  xy_mesh.block(0, 0, x_mesh.size(), 1) = x_mesh;
  xy_mesh.block(0, 1, y_mesh.size(), 1) = y_mesh;

  return xy_mesh;
}

template <typename Scalar = double>
Eigen::Matrix<Scalar, Eigen::Dynamic, 2>
meshgrid(const int x_size, const double x_start, const double x_end,
         const int y_size, const double y_start, const double y_end) {
  using Eigen::Map;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;

  // assume the following parameters:
  //
  // x_start = 0
  // x_end = 3
  // y_start = -2
  // y_end = 0

  // given those params x_range should be
  //
  // [0, 1, 2].transpose()
  const VectorXd x_range = VectorXd::LinSpaced(x_size, x_start, x_end);
  // y_range should be
  // [-2, -1]
  const RowVectorXd y_range = VectorXd::LinSpaced(y_size, y_start, y_end);

  // x_mesh would be
  // [0, 1, 2, 0, 1, 2].tranpose()
  MatrixXd x_mesh = x_range.replicate(y_range.size(), 1);
  // y_duplicated would be
  // [[-2, -1],
  //  [-2, -1],
  //  [-2, -2]]
  MatrixXd y_duplicated = y_range.replicate(x_range.size(), 1);
  // y_mesh is
  // [-2, -2, -2, -1, -1, -1].tranpose()
  Map<VectorXd> y_mesh(y_duplicated.data(), y_duplicated.size());

  // xy_mesh is
  // [[0, -2]
  //  [1, -2],
  //  [2, -2],
  //  [0, -1]
  //  [1, -1],
  //  [2, -1]]
  Matrix<Scalar, Eigen::Dynamic, 2> xy_mesh(x_mesh.size(), 2);
  xy_mesh.block(0, 0, x_mesh.size(), 1) = x_mesh;
  xy_mesh.block(0, 1, y_mesh.size(), 1) = y_mesh;

  return xy_mesh;
}

PluginUndistort::PluginUndistort(FrameBuffer *buffer,
                                 CameraParameters &camera_params)
    : VisionPlugin(buffer), camera_params(camera_params),
      settings(new VarList("Undistort")), enable(new VarBool("Enable", true)),
      interpolation_mode(new VarStringEnum("Interpolation Mode")) {

  settings->addChild(enable.get());
  settings->addChild(interpolation_mode.get());
}

ProcessResult PluginUndistort::process(FrameData *data,
                                       RenderOptions *options) {
  if (enable->getBool()) {

    if (undistort_map_x.empty() || undistort_map_y.empty()) {
      dbg("Creating undistortion map");
      makeUndistortMap(data->video.getWidth(), data->video.getHeight());

      // undistort_map_x.create(data->video.getHeight(), data->video.getWidth(),
      //                        CV_32FC1);
      // undistort_map_y.create(data->video.getHeight(), data->video.getWidth(),
      //                        CV_32FC1);
      // for (int r = 0; r < undistort_map_x.rows; r++) {
      //   for (int c = 0; c < undistort_map_x.cols; c++) {
      //     undistort_map_x.at<float>(r, c) = r;
      //     undistort_map_y.at<float>(r, c) = c;
      //   }
      // }
    }

    Image<rgb> *undistorted_image;
    if ((undistorted_image = reinterpret_cast<Image<rgb> *>(
             data->map.get("undistorted"))) == nullptr) {
      undistorted_image = reinterpret_cast<Image<rgb> *>(
          data->map.insert("undistorted", new Image<rgb>()));
    }
    undistorted_image->allocate(data->video.getWidth(),
                                data->video.getHeight());

    // TODO: This should be based on color format, but for now I'm assuming that
    // it is YUV422_UYVY;
    cv::Mat uyvy_image(data->video.getHeight(), data->video.getWidth(), CV_8UC2,
                       data->video.getData());
    cv::Mat distorted_image(data->video.getHeight(), data->video.getWidth(),
                            CV_8UC3);
    cv::cvtColor(uyvy_image, distorted_image, cv::COLOR_YUV2RGB_UYVY);

    cv::Mat undistorted_mat(undistorted_image->getHeight(),
                            undistorted_image->getWidth(), CV_8UC3,
                            undistorted_image->getData());
    // distorted_image.copyTo(undistorted_mat);

    cv::remap(distorted_image, undistorted_mat, undistort_map_x,
              undistort_map_y, cv::INTER_NEAREST);
  }
  return ProcessingOk;
}

VarList *PluginUndistort::getSettings() { return settings.get(); }

std::string PluginUndistort::getName() { return "Undistort"; }

void PluginUndistort::calculateUndistortedImageBounds(
    const int width, const int height, Eigen::Vector2d &upper_left,
    Eigen::Vector2d &lower_right) {
  // generate image coordinates
  //
  // This is similar to numpy.meshgrid. Should get ()width*height, 2)
  // matrix with all combinations of row,col indexes.
  //
  // [[0, 0],
  //  [1, 0],
  //  ...
  //  [width, 0],
  //  [0, 1],
  //  ...
  //  [width, 1],
  //  ...
  //  [0, height],
  //  ...
  //  [width, height]]
  const Eigen::Matrix<double, Eigen::Dynamic, 2> p_i = meshgrid(
      width, -width / 2.0, width / 2.0, height, -height / 2.0, height / 2.0);

  // undo scaling and offset to get distorted image coordinates
  const Eigen::RowVector2d principal_point{
      camera_params.principal_point_x->getDouble(),
      camera_params.principal_point_y->getDouble()};

  // subtract principal_point_x from first column, principal_point_y
  // from second column and divide all by focal length
  Eigen::Matrix<double, Eigen::Dynamic, 2> p_d =
      (p_i.rowwise() - principal_point) /
      camera_params.focal_length->getDouble();

  // undistort the coordinates
  const double distortion = camera_params.distortion->getDouble();
  const Eigen::VectorXd rd = p_d.rowwise().norm();
  const Eigen::VectorXd ru =
      rd.array() * (1 + rd.array() * rd.array() * distortion);
  p_d.rowwise().normalize();
  Eigen::Matrix<double, Eigen::Dynamic, 2> p_u =
      p_d.array().colwise() * ru.array();

  // find the minimum/maximum bounds of the new image
  upper_left = p_u.colwise().minCoeff();
  lower_right = p_u.colwise().maxCoeff();
}

template <typename Func> struct lambda_as_visitor_wrapper : Func {
  lambda_as_visitor_wrapper(const Func &f) : Func(f) {}
  template <typename S, typename I> void init(const S &v, I i, I j) {
    return Func::operator()(v, i, j);
  }
};

template <typename Mat, typename Func>
void visit_lambda(const Mat &m, const Func &f) {
  lambda_as_visitor_wrapper<Func> visitor(f);
  m.visit(visitor);
}

void PluginUndistort::makeUndistortMap(const int undistorted_width,
                                       const int undistorted_height) {
  Eigen::Vector2d upper_left{0, 0};
  Eigen::Vector2d lower_right{0, 0};
  calculateUndistortedImageBounds(undistorted_width, undistorted_height,
                                  upper_left, lower_right);

  const double focal_length = camera_params.focal_length->getDouble();
  const Eigen::RowVector2d principal_point{
      camera_params.principal_point_x->getDouble(),
      camera_params.principal_point_y->getDouble()};

  Eigen::Matrix<double, Eigen::Dynamic, 2> p_c =
      meshgrid(undistorted_width, -0.5, 0.5, undistorted_height, -0.5, 0.5);
  // Eigen::Matrix<double, Eigen::Dynamic, 2> p_c =
  //     meshgrid(undistorted_width, upper_left(0, 0), lower_right(0, 0),
  //              undistorted_height, upper_left(1, 0), lower_right(1, 0));

  // apply distortion
  const Eigen::VectorXd ru =
      p_c.rowwise().norm(); // treat each row as vector and compute length
  const double a = camera_params.distortion->getDouble();

  Eigen::VectorXd b =
      (-9.0 * a * a * ru.array()) +
      a * (a * (12.0 + (81.0 * a * ru.array() * ru.array()))).sqrt();

  // modify b based on sign of each value
  visit_lambda(b, [&b](double value, int i, int j) {
    b(i, j) = (value < 0.0) ? (-pow(value, 1.0 / 3.0)) : pow(value, 1.0 / 3.0);
  });

  Eigen::VectorXd rd = pow(2.0 / 3.0, 1.0 / 3.0) / b.array() -
                       b.array() / (pow(2.0 * 3.0 * 3.0, 1.0 / 3.0) * a);

  // normalize the coordinates in place
  // then scale by rd
  p_c.rowwise().normalize();
  Eigen::Matrix<double, Eigen::Dynamic, 2> p_d =
      p_c.array().colwise() * rd.array();

  Eigen::Matrix<double, Eigen::Dynamic, 2> p_i =
      (focal_length * p_d).rowwise() + principal_point;

  // convert these points to an opencv compatible remap mapping
  undistort_map_x.create(undistorted_height, undistorted_width, CV_32FC1);
  undistort_map_y.create(undistorted_height, undistorted_width, CV_32FC1);
  for (int r = 0; r < undistort_map_x.rows; r++) {
    for (int c = 0; c < undistort_map_x.cols; c++) {
      const int index = c * undistorted_width + r;
      undistort_map_x.at<float>(r, c) = p_i(index, 1);
      undistort_map_y.at<float>(r, c) = p_i(index, 0);
    }
  }
}
