# ─── GCS Bucket for tree data ────────────────────────────────────────────────

resource "google_storage_bucket" "genealogy_data" {
  name                        = "${var.project_id}-genealogy-data"
  location                    = var.region
  uniform_bucket_level_access = true
  storage_class               = "STANDARD"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
    }
    action {
      type = "Delete"
    }
  }
}

# ─── Artifact Registry ───────────────────────────────────────────────────────

resource "google_artifact_registry_repository" "genealogy_api" {
  repository_id = var.repository_name
  location      = var.region
  format        = "DOCKER"
  description   = "Genealogy API Docker images"
}

# ─── Cloud Run Service ───────────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "genealogy_api" {
  name                = var.service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run_sa.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}/${var.service_name}:latest"

      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.genealogy_data.name
      }

      env {
        name  = "JWT_SECRET"
        value = var.jwt_secret
      }

      env {
        name  = "SEED_USERS"
        value = var.seed_users
      }

      resources {
        limits = {
          memory = "512Mi"
          cpu    = "1"
        }
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    timeout = "60s"
  }

  depends_on = [google_artifact_registry_repository.genealogy_api]
}

# ─── Service Account for Cloud Run ───────────────────────────────────────────

resource "google_service_account" "cloud_run_sa" {
  account_id   = "genealogy-api-sa"
  display_name = "Genealogy API Cloud Run Service Account"
}

# Grant Cloud Run SA access to read/write GCS bucket
resource "google_storage_bucket_iam_member" "cloud_run_gcs" {
  bucket = google_storage_bucket.genealogy_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# ─── IAM: Allow public access to Cloud Run ───────────────────────────────────

resource "google_cloud_run_service_iam_binding" "public_invoker" {
  location = google_cloud_run_v2_service.genealogy_api.location
  project  = google_cloud_run_v2_service.genealogy_api.project
  service  = google_cloud_run_v2_service.genealogy_api.name
  role     = "roles/run.invoker"

  members = [
    "allUsers",
  ]
}

# ─── Cloud Build Trigger ─────────────────────────────────────────────────────

resource "google_cloudbuild_trigger" "genealogy_api_trigger" {
  name            = "genealogy-api-build"
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
    _SERVICE_NAME = var.service_name
    _REGION       = var.region
    _REPOSITORY   = var.repository_name
    _PROJECT_ID   = var.project_id
    _GCS_BUCKET   = google_storage_bucket.genealogy_data.name
  }
}
