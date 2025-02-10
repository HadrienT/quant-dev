variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Region for resources"
  type        = string
}

variable "bucket_name" {
  description = "Name of the bucket to store the function data code"
  type        = string
}
