#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     vector<Blob<Dtype>*>* top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  this->datum_channels_ = this->layer_param_.memory_data_param().channels();
  this->datum_height_ = this->layer_param_.memory_data_param().height();
  this->datum_width_ = this->layer_param_.memory_data_param().width();
  this->datum_size_ = this->datum_channels_ * this->datum_height_ *
      this->datum_width_;
  CHECK_GT(batch_size_ * this->datum_size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  (*top)[0]->Reshape(batch_size_, this->datum_channels_, this->datum_height_,
                     this->datum_width_);
  (*top)[1]->Reshape(batch_size_, 1, 1, 1);
  added_data_.Reshape(batch_size_, this->datum_channels_, this->datum_height_,
                      this->datum_width_);
  added_label_.Reshape(batch_size_, 1, 1, 1);
  data_ = NULL;
  labels_ = NULL;
  added_data_.cpu_data();
  added_label_.cpu_data();

  channel_scale_.clear();
  channel_mean_.clear();
  // read channel-wise mean and std
  if (this->layer_param_.memory_data_param().has_mean_file()) { 
    std::ifstream meanfile(this->layer_param_.memory_data_param().mean_file().c_str());
    if (!meanfile.is_open())  {
      LOG(FATAL) << "open mean_file in memory_data_param failed : " << this->layer_param_.memory_data_param().mean_file();
    }
    channel_scale_.resize(this->datum_channels_);
    channel_mean_.resize(this->datum_channels_);
    for (int i = 0; i != this->datum_channels_; ++i)  {
      meanfile >> channel_mean_[i] >> channel_scale_[i];
      channel_scale_[i] = 1.0 / channel_scale_[i]; 
      LOG(ERROR) << "channel " << i << 
        " : mean " << channel_mean_[i] << 
        " std " << 1. /channel_scale_[i];
    }
  }
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add Datum when earlier ones haven't been consumed"
      << " by the upper layers";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add";
  CHECK_LE(num, batch_size_) <<
      "The number of added datum must be no greater than the batch size";

  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = added_label_.mutable_cpu_data();
  if (this->mean_ != NULL && this->channel_mean_.size())  {
    LOG(FATAL) << "only one of transform_param and memory_data_param could have mean_file";
  }

  for (int batch_item_id = 0; batch_item_id < num; ++batch_item_id) {
    // Apply data transformations (mirror, scale, crop...)
    if (this->channel_mean_.size())  {
      this->data_transformer_.Transform(batch_item_id, datum_vector[batch_item_id], 
          this->channel_mean_, this->channel_scale_, top_data);
    } else  {
    this->data_transformer_.Transform(
        batch_item_id, datum_vector[batch_item_id], this->mean_, top_data);
    }
    top_label[batch_item_id] = datum_vector[batch_item_id].label();
  }
  // num_images == batch_size_
  Reset(top_data, top_label, batch_size_);
  has_new_data_ = true;
}


template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data_ = data;
  labels_ = labels;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";
  (*top)[0]->set_cpu_data(data_ + pos_ * this->datum_size_);
  (*top)[1]->set_cpu_data(labels_ + pos_);
  pos_ = (pos_ + batch_size_) % n_;
  has_new_data_ = false;
}

INSTANTIATE_CLASS(MemoryDataLayer);

}  // namespace caffe
