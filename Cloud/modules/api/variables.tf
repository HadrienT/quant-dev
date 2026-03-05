variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Region for Cloud Run"
  type        = string
}

variable "repository_name" {
  description = "Artifact Registry repository name (shared with webapp)"
  type        = string
}

variable "service_name" {
  description = "Cloud Run service name for the API"
  type        = string
  default     = "quantmodeling-api"
}

variable "github_owner" {
  description = "GitHub repository owner"
  type        = string
}

variable "github_repo_name" {
  description = "Name of the GitHub repository"
  type        = string
}

variable "builder_service_account" {
  description = "Service account used by Cloud Build"
  type        = string
}

variable "bq_project_id" {
  description = "BigQuery project ID"
  type        = string
  default     = ""
}

variable "bq_dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
  default     = ""
}

variable "bq_table_id" {
  description = "BigQuery table ID"
  type        = string
  default     = ""
}

variable "fred_api_key" {
  description = "API key for FRED"
  type        = string
  sensitive   = true
  default     = ""
}

variable "api_key" {
  description = "API key to protect the pricing service"
  type        = string
  sensitive   = true
  default     = ""
}
