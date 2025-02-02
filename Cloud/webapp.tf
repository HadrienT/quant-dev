resource "google_artifact_registry_repository" "repository" {
  repository_id = var.repository_name
  location      = var.region
  format        = "DOCKER"
  description   = "Artifact Registry for Cloud Run"
}

resource "google_cloudbuild_trigger" "build_trigger" {
  name            = "cloud-run-build-trigger"
  filename        = "Cloud/cloudbuild.yaml"
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
    _SERVICE_NAME = var.service_name
    _REGION       = var.region
    _REPOSITORY   = var.repository_name
    _PROJECT_ID   = var.project_id
  }
}

resource "google_cloud_run_v2_service" "service" {
  name                = var.service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}/${var.service_name}:latest"
      env {
        name  = "FRED_API_KEY"
        value = var.fred_api_key
      }

      resources {
        limits = {
          memory = "1024Mi"
          cpu    = "1"
        }
      }
    }
    vpc_access {
      connector = google_vpc_access_connector.cloudrun_connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    timeout = "600s"
  }
}


# resource "google_cloud_run_service_iam_policy" "no_auth" {
#   location = google_cloud_run_v2_service.service.location
#   project  = google_cloud_run_v2_service.service.project
#   service  = google_cloud_run_v2_service.service.name

#   policy_data = data.google_iam_policy.no_auth_policy.policy_data
# }


# data "google_iam_policy" "no_auth_policy" {
#   binding {
#     role = "roles/run.invoker"
#     members = [
#       "allUsers"
#     ]
#   }
# }


resource "google_cloud_run_domain_mapping" "portfolio_domain" {
  name     = var.domain_name
  location = var.region

  spec {
    route_name = google_cloud_run_v2_service.service.name
  }
  metadata {
    namespace = google_cloud_run_v2_service.service.project
  }
}


# data "google_iam_policy" "lb_only_policy" {
#   binding {
#     role = "roles/run.invoker"
#     members = [
#       "serviceAccount:${var.project_number}-compute@developer.gserviceaccount.com" # Load Balancer Service Account
#     ]
#   }
# }


# resource "google_cloud_run_service_iam_policy" "lb_only" {
#   location = google_cloud_run_v2_service.service.location
#   project  = google_cloud_run_v2_service.service.project
#   service  = google_cloud_run_v2_service.service.name

#   policy_data = data.google_iam_policy.lb_only_policy.policy_data
# }


resource "google_cloud_run_service_iam_binding" "lb_invoker" {
  location = google_cloud_run_v2_service.service.location
  project  = google_cloud_run_v2_service.service.project
  service  = google_cloud_run_v2_service.service.name
  role     = "roles/run.invoker"

  members = [
    "serviceAccount:${var.project_number}-compute@developer.gserviceaccount.com"
  ]
}


# resource "google_cloud_run_service_iam_member" "allow_all_users" {
#   location = google_cloud_run_v2_service.service.location
#   project  = google_cloud_run_v2_service.service.project
#   service  = google_cloud_run_v2_service.service.name
#   role     = "roles/run.invoker"
#   member   = "allUsers"
# }
