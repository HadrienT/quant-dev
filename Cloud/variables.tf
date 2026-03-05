variable "project_id" {
  description = "The ID of the project in which resources will be created"
  type        = string
}

variable "project_number" {
  description = "The number of the project in which resources will be created"
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

variable "api_service_name" {
  description = "Cloud Run service name for the pricing API"
  type        = string
  default     = "quantmodeling-api"
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

variable "api_key" {
  description = "API key to protect the pricing service"
  type        = string
  sensitive   = true
  default     = ""
}

variable "enable_webapp" {
  description = "Enable or disable webapp infrastructure (Cloud Run and related resources)"
  type        = bool
  default     = true
}

variable "enable_genealogy" {
  description = "Enable or disable genealogy API infrastructure (Cloud Run, GCS, Cloud Build)"
  type        = bool
  default     = false
}

variable "genealogy_repository_name" {
  description = "Artifact Registry repository name for genealogy API"
  type        = string
  default     = "genealogy-api"
}

variable "genealogy_service_name" {
  description = "Cloud Run service name for genealogy API"
  type        = string
  default     = "genealogy-api"
}

variable "genealogy_github_repo_name" {
  description = "GitHub repository name for genealogy project"
  type        = string
  default     = "genealogy"
}

variable "genealogy_jwt_secret" {
  description = "JWT secret key for genealogy API authentication"
  type        = string
  sensitive   = true
}

variable "genealogy_seed_users" {
  description = "JSON array of seed users for genealogy API: [{username, password, role}]"
  type        = string
  sensitive   = true
  default     = "[]"
}

variable "bq_reader_key_path" {
  description = "Output path for the BigQuery reader key JSON"
  type        = string
  default     = "./secrets/bq-key.json"
}
