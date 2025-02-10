variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Region for resources"
  type        = string
}

variable "zone" {
  description = "Zone for the VM"
  type        = string
}

variable "vm_name" {
  description = "Name of the development VM"
  type        = string
}

variable "ssh_key" {
  description = "SSH key to inject into the VM"
  type        = string
}

variable "subnetwork" {
  description = "Subnetwork in which the VM will be created"
  type        = string
}
