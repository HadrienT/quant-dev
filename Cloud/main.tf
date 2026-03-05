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

module "bq_access" {
  source          = "./modules/bqAccess"
  project_id      = var.project_id
  key_output_path = var.bq_reader_key_path
}

module "api" {
  count                   = var.enable_webapp ? 1 : 0
  source                  = "./modules/api"
  project_id              = var.project_id
  region                  = var.region
  repository_name         = var.repository_name
  service_name            = var.api_service_name
  github_owner            = var.github_owner
  github_repo_name        = var.github_repo_name
  builder_service_account = var.builder_service_account
  bq_project_id           = var.bq_project_id
  bq_dataset_id           = var.bq_dataset_id
  bq_table_id             = var.bq_table_id
  fred_api_key            = var.fred_api_key
  api_key                 = var.api_key
}

module "webapp" {
  count                   = var.enable_webapp ? 1 : 0
  source                  = "./modules/webapp"
  project_id              = var.project_id
  project_number          = var.project_number
  region                  = var.region
  repository_name         = var.repository_name
  service_name            = var.service_name
  github_owner            = var.github_owner
  github_repo_name        = var.github_repo_name
  builder_service_account = var.builder_service_account
  api_url                 = module.api[0].api_service_url
  domain_name             = var.domain_name
}

module "genealogy" {
  count                   = var.enable_genealogy ? 1 : 0
  source                  = "./modules/genealogy"
  project_id              = var.project_id
  region                  = var.region
  repository_name         = var.genealogy_repository_name
  service_name            = var.genealogy_service_name
  github_owner            = var.github_owner
  github_repo_name        = var.genealogy_github_repo_name
  builder_service_account = var.builder_service_account
  jwt_secret              = var.genealogy_jwt_secret
  seed_users              = var.genealogy_seed_users
}
