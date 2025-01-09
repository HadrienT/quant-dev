variable "project_id" {
  description = "The ID of the project in which resources will be created"
  type        = string
}

variable "region" {
  description = "The region where resources will be created"
  type        = string
  default     = "europe-west1"
}

variable "zone" {
  description = "The zone where resources will be created"
  type        = string
}

variable "tf_service_account" {
  description = "The email of the service account to impersonate"
  type        = string
}

variable "bucket_data_collector_code" {
  description = "Name of the bucket to store the data collector source code"
  type        = string
  default = "bucket_data_collector_code"
}
