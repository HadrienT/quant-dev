variable "project_id" {
  description = "The ID of the project in which resources will be created"
  type        = string
}

variable "region" {
  description = "The region where resources will be created"
  type        = string
}

variable "zone" {
  description = "The zone where resources will be created"
  type        = string
}

variable "tf_service_account" {
  description = "The email of the service account to impersonate"
  type        = string
}
