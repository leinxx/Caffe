#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const vector<Dtype>& mean,
                                       const vector<Dtype>& scale,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();
  const int size_one_channel = datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const bool rotate = param_.rotate();
  CHECK_EQ(mean.size(), scale.size());

  Blob<Dtype> buffer;
  buffer.Reshape(1, channels, height, width);
  Dtype* buffer_data = buffer.mutable_cpu_data();
  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }
  
  if (rotate && phase_ == Caffe::TRAIN) {
    cv::Mat src, dst;
    src = cv::Mat::zeros(datum.height(), datum.width(), CV_8UC1); 
    cv::Point2f pt(src.cols/2., src.rows/2.);
    int angle = Rand() % 360;
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    for (int c = 0; c != channels; ++c) {
      for (int data_index = 0; data_index != datum.height() * datum.width() ; ++data_index )  {
        Dtype datum_element =
          static_cast<Dtype>(static_cast<uint8_t>(data[data_index + c * height * width]));
        src.data[data_index] = datum_element;
      }
      cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
      for (int data_index = 0; data_index != datum.height() * datum.width() ; ++data_index )  {
        buffer_data[data_index + c * height * width] = dst.data[data_index];
      }

      //cv::imwrite("before.png", src);
      //cv::imwrite("after.png", dst);
      
      //cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
      //for (int data_index = 0; data_index != size_one_channel; ++data_index)  {
      //  out.at<unsigned char>(data_index / width, data_index % width) = buffer_data[data_index + c * height * width];
      //}
      //cv::imwrite("out.tif",out);
      
    }
  } else  {
      for (int data_index = 0; data_index != size ; ++data_index )  {
        unsigned char datum_element =
          static_cast<uint8_t>(data[data_index]);
        buffer_data[data_index] = datum_element;
        if (datum_element < 0 || datum_element > 255) {
          LOG(FATAL) << "asdasd";
        }
      }
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element = buffer_data[data_index];
            transformed_data[top_index] =
                (datum_element - mean[c]) * scale[c];
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element = 
                buffer_data[data_index];
            transformed_data[top_index] =
                (datum_element - mean[c]) * scale[c];
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int c = 0; c < channels; ++c) {
        int offset = c * size_one_channel + batch_item_id * size;
        for (int j = 0; j <width * height ; ++j) {
          Dtype datum_element =
            buffer_data[c * size_one_channel + j ];
          transformed_data[j + offset] =
            (datum_element - mean[c]) * scale[c];
        }
      }
    } else {
      LOG(ERROR) << "This is not tested. The input data type is not correct";
      for (int c = 0; c < channels; ++c) {
        int offset = c * size_one_channel + batch_item_id * size;
        for (int j = 0; j < width * height; ++j) {
          transformed_data[j + offset] =
            (datum.float_data(j + c * size_one_channel) - mean[c]) * scale[c];
        }
      }
    }
  }

}
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size() || param_.rotate());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
