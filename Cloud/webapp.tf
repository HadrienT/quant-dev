resource "google_artifact_registry_repository" "repository" {
  repository_id = var.repository_name
  name         = var.repository_name
  location     = var.region
  format       = "DOCKER"
  description  = "Artifact Registry for Cloud Run"
}

resource "google_cloudbuild_trigger" "build_trigger" {
  name = "cloud-run-build"

  filename = "cloudbuild.yaml"
  substitutions = {
    _SERVICE_NAME   = var.service_name
    _REGION         = var.region
    _REPOSITORY     = var.repository_name
    _PROJECT_ID     = var.project_id
  }
}

resource "google_cloud_run_service" "service" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      containers {
        image = "${google_artifact_registry_repository.repository.repository_url}/${var.service_name}:latest"
        resources {
          limits = {
            memory = "512Mi"
            cpu    = "1"
          }
        }
      }
    }
  }
}

resource "google_cloud_run_service_iam_policy" "no_auth" {
  location = google_cloud_run_service.service.location
  project  = google_cloud_run_service.service.project
  service  = google_cloud_run_service.service.name

  policy_data = data.google_iam_policy.no_auth_policy.policy_data
}

data "google_iam_policy" "no_auth_policy" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers"
    ]
  }
}


resource "google_cloudbuild_trigger" "github_trigger" {
  name = "github-cloud-build-trigger"

  github {
    owner       = var.github_owner     
    name        = var.github_repo_name  
    push {
      branch = "^main$"                 # Watch for changes on the main branch
    }
  }

  filename = "cloudbuild.yaml"

  substitutions = {
    _SERVICE_NAME   = var.service_name
    _REGION         = var.region
    _PROJECT_ID     = var.project_id
    _REPOSITORY     = var.repository_name
  }
}
