#############################
# Static IP for VM
#############################
resource "google_compute_address" "static_ip" {
  name   = "${var.vm_name}-static-ip"
  region = "europe-west3"
}

#############################
# (VM) Compute Engine Instance
#############################
resource "google_compute_instance" "instance" {
  name         = var.vm_name
  machine_type = "e2-standard-2"
  zone         = "europe-west3-a"

  boot_disk {
    auto_delete = true
    device_name = "${var.vm_name}-dsk"

    initialize_params {
      image = "projects/debian-cloud/global/images/debian-12-bookworm-v20241210"
      size  = 100
      type  = "pd-standard"
    }
  }
  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false


  labels = {
    goog-ec-src = "vm_add-tf"
  }

  network_interface {
    subnetwork = "projects/quant-dev-442615/regions/europe-west3/subnetworks/default"

    access_config {
      network_tier = "PREMIUM"
      nat_ip       = google_compute_address.static_ip.address
    }
  }
  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
  }

  service_account {
    email = "956975863647-compute@developer.gserviceaccount.com"
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append"
    ]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  metadata = {
    ssh-keys = "${var.vm_name}:${var.ssh_key}"
  }
}
