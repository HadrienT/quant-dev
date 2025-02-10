variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "project_number" {
  description = "GCP project number"
  type        = string
}

variable "region" {
  description = "Region for Cloud Run"
  type        = string
}

variable "repository_name" {
  description = "Name of the Artifact Registry repository"
  type        = string
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
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

variable "fred_api_key" {
  description = "API key for FRED"
  type        = string
  sensitive   = true
}

variable "domain_name" {
  description = "Custom domain name for Cloud Run"
  type        = string
}
