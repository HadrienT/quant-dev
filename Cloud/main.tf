provider "google" {
  project                     = var.project_id
  region                      = var.region
  zone                        = var.zone
  impersonate_service_account = var.tf_service_account
}

module "vm" {
  source     = "./modules/vm"
  project_id = var.project_id
  region     = var.region
  zone       = var.zone
  vm_name    = var.vm_name
  ssh_key    = var.ssh_key
  subnetwork = "projects/${var.project_id}/regions/${var.region}/subnetworks/default"
}

module "dataCollector" {
  source      = "./modules/dataCollector"
  project_id  = var.project_id
  region      = var.region
  bucket_name = var.bucket_data_collector_code
}

module "webapp" {
  source                  = "./modules/webapp"
  project_id              = var.project_id
  project_number          = var.project_number
  region                  = var.region
  repository_name         = var.repository_name
  service_name            = var.service_name
  github_owner            = var.github_owner
  github_repo_name        = var.github_repo_name
  builder_service_account = var.builder_service_account
  fred_api_key            = var.fred_api_key
  domain_name             = var.domain_name
}
