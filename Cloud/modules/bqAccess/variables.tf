variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "service_account_id" {
  description = "Service account id for BigQuery reader"
  type        = string
  default     = "bq-reader"
}

variable "service_account_display_name" {
  description = "Display name for the BigQuery reader service account"
  type        = string
  default     = "BigQuery Reader"
}

variable "key_output_path" {
  description = "Output path for the BigQuery reader key JSON"
  type        = string
}

variable "portfolio_bucket_name" {
  description = "GCS bucket name for portfolio storage"
  type        = string
  default     = "quant-portfolios"
}

variable "portfolio_bucket_location" {
  description = "GCS bucket location"
  type        = string
  default     = "EU"
}
