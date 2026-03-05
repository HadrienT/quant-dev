variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "repository_name" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "genealogy-api"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "genealogy-api"
}

variable "github_owner" {
  description = "GitHub repository owner"
  type        = string
}

variable "github_repo_name" {
  description = "GitHub repository name"
  type        = string
  default     = "genealogy"
}

variable "builder_service_account" {
  description = "Service account used by Cloud Build"
  type        = string
}

variable "jwt_secret" {
  description = "Secret key for JWT token signing"
  type        = string
  sensitive   = true
}

variable "seed_users" {
  description = "JSON array of initial users: [{username, password, role}]"
  type        = string
  sensitive   = true
  default     = "[]"
}
