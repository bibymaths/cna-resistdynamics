<!-- SPDX-FileCopyrightText: 2025 Abhinav Mishra -->
<!-- SPDX-License-Identifier: MIT -->

# API Reference

This page documents the public Python API of the `tumorfits` package.

---

## tumorfits.dataio

::: tumorfits.dataio
    options:
      show_root_heading: true
      members:
        - export_all_patient_data
        - PATIENT_ID_PATTERN

---

## tumorfits.odeio

::: tumorfits.odeio
    options:
      show_root_heading: true
      members:
        - PatientData
        - load_patient_data
        - load_sample_list
        - get_patients_with_flag
        - load_drivers

---

## tumorfits.odemodel

::: tumorfits.odemodel
    options:
      show_root_heading: true

---

## tumorfits.oderunner

::: tumorfits.oderunner
    options:
      show_root_heading: true
      members:
        - ODEFitConfig
        - fit_ode
        - fit_ode_cohort
        - run_ode_cli

---

## tumorfits.pdemodel

::: tumorfits.pdemodel
    options:
      show_root_heading: true
      members:
        - PDEConfig

---

## tumorfits.pderunner

::: tumorfits.pderunner
    options:
      show_root_heading: true

---

## tumorfits.meshview

::: tumorfits.meshview
    options:
      show_root_heading: true
      members:
        - load_all_patient_params
        - run_cancer_simulation_2d
        - plot_resistance_zones
        - plot_growth_streamlines
        - plot_drug_efficacy
        - run_mesh_view_pipeline

---

## tumorfits.metrics

::: tumorfits.metrics
    options:
      show_root_heading: true

---

## tumorfits.utils

::: tumorfits.utils
    options:
      show_root_heading: true

---

## tumorfits.timelog

::: tumorfits.timelog
    options:
      show_root_heading: true
