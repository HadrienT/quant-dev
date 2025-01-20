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
  default     = "bucket_data_collector_code"
}


### VM
variable "ssh_key" {
  description = "SSH public key for the instance"
  type        = string
}

variable "vm_name" {
  default = "ht-vm-e2-medium"
}

### Webapp.tf
variable "repository_name" {
  description = "Artifact Registry repository name"
  type        = string
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
}


variable "github_owner" {
  description = "Propriétaire du dépôt GitHub (utilisateur ou organisation)"
  type        = string
}

variable "github_repo_name" {
  description = "Nom du dépôt GitHub"
  type        = string
}

variable "builder_service_account" {
  description = "The email of the service account to impersonate"
  type        = string
}

variable "fred_api_key" {
  description = "API Key for FRED"
  type        = string
  sensitive   = true
}

variable "domain_name" {
  description = "Custom domain to map to Cloud Run service"
}
