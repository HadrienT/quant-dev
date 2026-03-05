#############################
# Cloud Build Trigger
#############################
resource "google_cloudbuild_trigger" "build_trigger" {
  name            = "cloud-run-build-trigger"
  filename        = "cloudbuild.yaml"
  location        = "global"
  service_account = "projects/${var.project_id}/serviceAccounts/${var.builder_service_account}"

  github {
    owner = var.github_owner
    name  = var.github_repo_name
    push {
      branch = "^main$"
    }
  }

  substitutions = {
    _SERVICE_NAME     = var.service_name
    _API_SERVICE_NAME = "quantmodeling-api"
    _REGION           = var.region
    _REPOSITORY       = var.repository_name
    _PROJECT_ID       = var.project_id
    _API_URL          = var.api_url
  }
}
