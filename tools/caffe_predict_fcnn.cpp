#include <glog/logging.h>
#include <gflags/gflags.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include <boost/algorithm/string.hpp>
#include "caffe/util/upgrade_proto.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using caffe::NetParameter;

DEFINE_string(image, "", "Input images, file pathes of different bands of the same image need to be seperated by comma. example: b1.tif,b2.tif ");
DEFINE_string(model, "", "model path");
DEFINE_string(weights, "", "model deploy architecture file");
DEFINE_string(predict, "", "Predict path");
DEFINE_string(meanfile, "", "mean file");
DEFINE_string(featurename, "", "feature name");

void fill_predict(cv::Mat& predict, int roff, int coff, shared_ptr<Blob<float> > src, const float scale) {
// fill src data to a roi of predict defined bu roff, coff, rows and cols, start index is the index in the roi
// n : number of src data to fill
  const float* src_data = src->cpu_data();
  int rend = std::min(predict.rows, src->height() + roff);
  int cend = std::min(predict.cols, src->width() + coff);
  for (int r = roff; r != rend; ++r) {
    for (int c = coff; c != cend; ++c) {
      predict.at<unsigned char>(r, c) = src_data[(r - roff) * src->width() + c - coff] * scale;
    }
  }
}

void copy_from_mat(cv::Mat& image, int roff, int coff, int rows, int cols, float mean, float scale, float* dst)  {
  int rend = std::min(roff + rows, image.rows); 
  int cend = std::min(coff + cols, image.cols); 
  for (int i = roff; i != rend; ++i) {
    for (int j = coff; j != cend; ++j)  {
      dst[(i-roff) * cols + j - coff] = (static_cast<float>(image.at<unsigned char>(i, j)) - mean) * scale;
    }
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK_NE(FLAGS_image, "");
  CHECK_NE(FLAGS_model, "");
  CHECK_NE(FLAGS_predict, "");
  CHECK_NE(FLAGS_weights, "");
  CHECK_NE(FLAGS_featurename, "");
  CHECK_NE(FLAGS_meanfile, "");

  // network initialization
  Caffe::set_phase(Caffe::TEST);
  caffe::NetParameter param;
  ReadNetParamsFromTextFileOrDie(FLAGS_model, &param);
  CHECK_EQ(param.input_dim_size(), 4);
  
  //caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  // split mutiple input image file path
  std::vector<std::string> image_files;
  boost::split(image_files, FLAGS_image, boost::is_any_of(","));
  for (int i = 0; i != image_files.size(); ++i) {
    LOG(ERROR) << "band : " << image_files[i];
  }
  // read image files and get patches
  vector<cv::Mat> image_bands;
  int channels = 0;
  for (int band = 0; band != image_files.size(); ++band) {
    image_bands.push_back(cv::imread(image_files[band], -1));
    if (band != 0)  {
      CHECK_EQ(image_bands[0].rows, image_bands[band].rows) << "input images should be the same dimension";
      CHECK_EQ(image_bands[0].cols, image_bands[band].cols) << "input images should be the same dimension";
    }
    channels += image_bands[band].channels();
  }
  const int W = image_bands[0].cols;
  const int H = image_bands[0].rows;

  param.set_input_dim(0, 1);
  param.set_input_dim(1, channels);
  param.set_input_dim(2, 500);
  param.set_input_dim(3, 500);
  Net<float> caffe_net(param);

  vector<float> mean;
  vector<float> scale;
  LOG(ERROR) << FLAGS_meanfile;
  std::ifstream meanfile(FLAGS_meanfile.c_str());
  scale.resize(channels);
  mean.resize(channels);
  for (int i = 0; i != channels; ++i)  {
    meanfile >> mean[i] >> scale[i];
    scale[i] = 1.0 / scale[i]; 
    LOG(ERROR) << "channel " << i << 
      " : mean " << mean[i] << 
      " std " << 1. / scale[i];
  }

  vector<Blob<float>* > input = caffe_net.input_blobs();
  CHECK_EQ(input.size(), 1);
  Blob<float>* input_data = input[0];
  const int width = input_data->width();
  const int height = input_data->height();
  CHECK_EQ(channels, input_data->channels());
  const int roff = 20;
  const int coff = 20;
  cv::Mat predict = cv::Mat::zeros(H, W, CV_8U);
        cv::imwrite(FLAGS_predict, predict);
  int rstep = 450;
  int cstep = 450;
  for (int r = roff; r <= H-20; r += rstep) {
    for (int c = coff; c <= W-20; c += cstep)  {
      // extract patch
      float* data = input_data->mutable_cpu_data() ;
      for (int b = 0; b != channels; ++b, data += width * height)  {
        copy_from_mat(image_bands[b], r - roff, c - coff, height, width, mean[b], scale[b], data);
      }
        cv::imwrite(FLAGS_predict, predict);
        // predict
        LOG(ERROR) << r << " " << c;
        caffe_net.ForwardPrefilled();
        const shared_ptr<Blob<float> > feature_blob = caffe_net.blob_by_name(FLAGS_featurename);
        cv::imwrite(FLAGS_predict, predict);
        fill_predict(predict, r, c, feature_blob, 100);
        cv::imwrite(FLAGS_predict, predict);
    }
  }

  cv::imwrite(FLAGS_predict, predict);
}

