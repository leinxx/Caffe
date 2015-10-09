#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h>
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
namespace caffe {
  template <typename T>
    string NumberToString ( T Number )
    {
      ostringstream ss;
      ss << Number;
      return ss.str();
    }


template <typename Dtype>
DatumDataLayer<Dtype>::~DatumDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void DatumDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Read the file with filenames
  const string& source = this->layer_param_.datum_data_param().source();
  CHECK_GT(source.size(), 0);
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good())
      << "Could not open datum file list (filename: \""+ source + "\")";
  string folder = source.substr(0, source.rfind('/') + 1);
  string filename;
  while (infile >> filename) {
    lines_.push_back(folder + filename);
  }

  if (this->layer_param_.datum_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  CHECK(!lines_.empty())
      << "Image list is empty (filename: \"" + source + "\")";
  // Read a data point, and use it to initialize the top blob.
  current_idx_ = 0;
  update_prefetch_buffer();
  Datum datum = prefetch_buffer_[0];
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.datum_data_param().batch_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();

  // read scale and mean
  std::ifstream meanfile(this->layer_param_.datum_data_param().mean_file().c_str());
  scale_.resize(this->datum_channels_);
  mean_.resize(this->datum_channels_);
  for (int i = 0; i != this->datum_channels_; ++i)  {
    meanfile >> mean_[i] >> scale_[i];
    scale_[i] = 1.0 / scale_[i]; 
    LOG(ERROR) << "channel " << i << " : mean " << mean_[i] << " std " << scale_[i];
  }

}

template <typename Dtype>
void DatumDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void DatumDataLayer<Dtype>::update_prefetch_buffer() {
  // read two files and shuffle, so each batch have the chance to contain patches from multiple images
  prefetch_buffer_.clear();
  for (int i = 0; i < 2; ++i) {
    if (lines_id_ >= lines_.size())  {
      // ShuffleImages();
      lines_id_ = 0;
    }
    LOG(ERROR) << lines_[lines_id_] << " : " << lines_id_; 
    DatumVector buffer;
    ReadProtoFromBinaryFileOrDie(lines_[lines_id_], &buffer);
    for (int idx = 0; idx != buffer.datums_size(); ++idx) {
      prefetch_buffer_.push_back(buffer.datums(idx));
    }
    lines_id_++;
  }
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(prefetch_buffer_.begin(), prefetch_buffer_.end(), prefetch_rng);
  current_idx_ = 0;
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DatumDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  DatumDataParameter datum_data_param = this->layer_param_.datum_data_param();
  const int batch_size = datum_data_param.batch_size();
  const Dtype label_scale = datum_data_param.label_scale();
  // datum scales
  for (int idx = 0; idx != batch_size; ++idx) {
    if (current_idx_ == prefetch_buffer_.size() || prefetch_buffer_.size() == 0) {
      update_prefetch_buffer();
      // LOG(ERROR) << "update_bufer_size: " << prefetch_buffer_.size();
    }
    datum = prefetch_buffer_[current_idx_];
    // write_datum_to_image(datum, idx);
    current_idx_++;
    
    // Apply transformations (mirror, crop...) to the data
    this->data_transformer_.Transform(idx, datum, this->mean_, this->scale_, top_data);

    top_label[idx] = Dtype(datum.label()) * label_scale;
  }
  // this->prefetch_data_.print_stats();
  // this->prefetch_label_.print_stats();

  // write_blob_to_image(this->prefetch_data_);
}

template <typename Dtype>
void DatumDataLayer<Dtype>::write_blob_to_image(Blob<Dtype>& blob)  {
  const int N = blob.num();
  const int C = blob.channels();
  const int W = blob.width();
  const int H = blob.height();
  for (int n = 0; n != N; ++n)  {
    for (int c = 0; c != C; ++c)  {
      std::string filename = NumberToString(n) + string("_") + string(NumberToString(c)) + string("blob.png");
      cv::Mat mat = cv::Mat::zeros(H, W, CV_8U);
      int offset = n * W * C * H + c * W * H;
      for (int i = 0; i != H; ++i)  {
        for (int j = 0; j != W; ++j, ++offset)  {
          mat.at<unsigned char>(i,j) = blob.cpu_data()[offset] / scale_[c] + mean_[c] ;
        }
      }
      cv::imwrite(filename, mat);
    }
  }
}

template <typename Dtype>
void DatumDataLayer<Dtype>::write_datum_to_image(Datum& datum, int idx)  {
  const int C = datum.channels();
  const int W = datum.width();
  const int H = datum.height();
  for (int c = 0; c != C; ++c)  {
    std::string filename = string(NumberToString(idx)) + string("_") + string(NumberToString(c)) + string("datum.png");
    cv::Mat mat = cv::Mat::zeros(H, W, CV_8U);
    int offset =  c * W * H;
    for (int i = 0; i != H; ++i)  {
      for (int j = 0; j != W; ++j, ++offset)  {
        mat.at<unsigned char>(i,j) = static_cast<uint8_t>(datum.data()[offset]);
      }
    }
    cv::imwrite(filename, mat);
  }
}
INSTANTIATE_CLASS(DatumDataLayer);

}  // namespace caffe
