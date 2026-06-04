pkgs <- c("readxl", "dplyr", "tidyr", "mgcv", "ggplot2", "patchwork", "scales")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit <- grep("--file=", args, fixed = TRUE, value = TRUE)
  if (length(hit) > 0) return(dirname(normalizePath(sub("--file=", "", hit[1]), winslash = "/")))
  normalizePath(getwd(), winslash = "/")
}

script_dir <- get_script_dir()
workflow_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
bundle_path <- file.path(workflow_dir, "2_Train+Test", "output", "final_model_bundle_chla.rds")
future_dir <- file.path(workflow_dir, "1_Data", "Future", "1_merged")
zone_path <- list.files(script_dir, pattern = "\\.xlsx$", full.names = TRUE)[1]
out_dir <- file.path(script_dir, "output")
trib_out_dir <- file.path(script_dir, "output_tributaries")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(trib_out_dir, recursive = TRUE, showWarnings = FALSE)

tnr <- "Times New Roman"
base_font_size <- 18
ssp_colors <- c(ssp245 = "#2196F3", ssp585 = "#F44336")
ssp_labels <- c(ssp245 = "SSP2-4.5", ssp585 = "SSP5-8.5")

pt_to_mm <- function(pt) pt / 2.845276

safe_write_csv <- function(x, file) {
  tryCatch(write.csv(x, file, row.names = FALSE),
           error = function(e) warning("Could not write ", file, ": ", conditionMessage(e), call. = FALSE))
}

save_plot <- function(plot, base_path, width = 12, height = 10) {
  ggsave(paste0(base_path, ".pdf"), plot, width = width, height = height, device = cairo_pdf)
  ggsave(paste0(base_path, ".png"), plot, width = width, height = height,
         dpi = 320, device = grDevices::png, type = "cairo")
}

standardize_basin_names <- function(x) {
  dplyr::recode(x, "ModaoRiver" = "MoDaoRiver", "YulinRiver" = "YuLinRiver", .default = x)
}

nice_lim <- function(vals, n = 5) {
  finite_vals <- vals[is.finite(vals)]
  if (length(finite_vals) == 0) return(list(limits = c(0, 1), breaks = c(0, 0.5, 1)))
  brks <- pretty(finite_vals, n = n)
  list(limits = range(brks), breaks = brks)
}

make_trend_label <- function(df, y_col) {
  df <- df |> filter(is.finite(.data[[y_col]]), is.finite(year))
  if (nrow(df) < 3 || length(unique(df$year)) < 2) {
    return("atop(slope == NA, atop(R^2 == NA, P == NA))")
  }
  fit <- lm(stats::as.formula(paste(y_col, "~ year")), data = df)
  slope <- coef(fit)[2]
  r2 <- summary(fit)$r.squared
  pval <- summary(fit)$coefficients[2, 4]
  p_expr <- ifelse(pval < 0.001, "P < 0.001", sprintf("P == %.3f", pval))
  sprintf("atop(slope == %.4f, atop(R^2 == %.2f, %s))", slope, r2, p_expr)
}

summarise_ensemble <- function(data, group_col) {
  raw_agg <- data |>
    group_by(.data[[group_col]], year, model, scenario) |>
    summarise(chla = mean(chla_pred, na.rm = TRUE), .groups = "drop")

  ensemble <- raw_agg |>
    group_by(.data[[group_col]], year, scenario) |>
    summarise(
      chla_mean = mean(chla, na.rm = TRUE),
      chla_sd = sd(chla, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      chla_lo = chla_mean - chla_sd,
      chla_hi = chla_mean + chla_sd
    )

  list(raw_agg = raw_agg, ensemble = ensemble)
}

plot_one_panel <- function(data, group_col, group_name, panel_tag, ylab,
                           font_size = base_font_size) {
  df <- data |>
    filter(.data[[group_col]] == group_name) |>
    mutate(scenario = factor(scenario, levels = names(ssp_colors)))
  y_ax <- nice_lim(c(df$chla_lo, df$chla_hi))
  lbl_245 <- make_trend_label(df |> filter(scenario == "ssp245"), "chla_mean")
  lbl_585 <- make_trend_label(df |> filter(scenario == "ssp585"), "chla_mean")
  x_left <- 2025 + (2100 - 2025) * 0.03
  x_right <- 2025 + (2100 - 2025) * 0.44

  ggplot(df, aes(x = year, color = scenario, fill = scenario)) +
    geom_ribbon(aes(ymin = chla_lo, ymax = chla_hi), alpha = 0.2, color = NA) +
    geom_line(aes(y = chla_mean), linewidth = 0.8) +
    annotate("text", x = c(x_left, x_right), y = c(Inf, Inf),
             label = c(lbl_245, lbl_585), color = unname(ssp_colors[c("ssp245", "ssp585")]),
             hjust = 0, vjust = 1.15, size = pt_to_mm(font_size), family = tnr, parse = TRUE) +
    annotate("text", x = 2100, y = Inf, label = panel_tag, hjust = 1.1, vjust = 1.3,
             size = pt_to_mm(font_size), fontface = "bold", family = tnr, color = "black") +
    scale_color_manual(values = ssp_colors, labels = ssp_labels) +
    scale_fill_manual(values = ssp_colors, labels = ssp_labels) +
    scale_x_continuous(limits = c(2025, 2100), breaks = seq(2025, 2100, by = 25), expand = expansion(0)) +
    scale_y_continuous(breaks = y_ax$breaks, expand = expansion(0)) +
    coord_cartesian(ylim = c(y_ax$limits[1], y_ax$limits[2] + 0.20 * diff(y_ax$limits)), expand = FALSE) +
    labs(title = group_name, x = NULL, y = ylab, color = "Scenario", fill = "Scenario") +
    theme_bw(base_size = font_size, base_family = tnr) +
    theme(
      plot.title = element_text(face = "bold", size = font_size, family = tnr, hjust = 1),
      axis.title.y = element_text(size = font_size, family = tnr),
      axis.text = element_text(size = font_size, family = tnr),
      legend.position = "none",
      panel.grid.minor = element_blank()
    )
}

make_4panel <- function(data, group_col, group_order, ylab, fname, plot_dir,
                        font_size = base_font_size) {
  panel_tags <- c("(a)", "(b)", "(c)", "(d)")
  plots <- lapply(seq_along(group_order), function(i) {
    plot_one_panel(data, group_col, group_order[i], panel_tags[i], ylab, font_size) +
      theme(
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.text = element_text(size = font_size, family = tnr),
        legend.title = element_text(size = font_size, family = tnr)
      )
  })
  combined <- ((plots[[1]] | plots[[2]]) / (plots[[3]] | plots[[4]]) / guide_area()) +
    plot_layout(guides = "collect", heights = c(1, 1, 0.08))
  save_plot(combined, file.path(plot_dir, fname))
  cat("Saved plot:", file.path(plot_dir, fname), ".pdf/.png\n")
}

cat("========== 1. Load CHLA model bundle and zone table ==========\n")
if (!file.exists(bundle_path)) stop("Model bundle not found: ", bundle_path)
bundle <- readRDS(bundle_path)
if (!identical(bundle$response_vars, "chla")) stop("Expected CHLA model bundle.")

if (length(zone_path) == 0 || is.na(zone_path)) stop("Zone xlsx not found in ", script_dir)
zone_raw <- readxl::read_excel(zone_path)
if (!all(c("basin_name") %in% names(zone_raw)) || ncol(zone_raw) < 2) {
  stop("Zone table must contain basin_name and one zone column.")
}
zone_values <- unique(zone_raw[[2]])
if (length(zone_values) != 3) stop("Expected three zone labels; found ", length(zone_values), ".")
zone_df <- zone_raw |>
  transmute(basin_name = standardize_basin_names(basin_name), zone_raw = .data[[names(zone_raw)[2]]]) |>
  mutate(zone = case_when(
    zone_raw == zone_values[1] ~ "Lower TGR",
    zone_raw == zone_values[2] ~ "Middle TGR",
    zone_raw == zone_values[3] ~ "Upper TGR",
    TRUE ~ NA_character_
  )) |>
  select(basin_name, zone)

cat("========== 2. Load future predictors from 1_Data/Future ==========\n")
future_files <- list.files(future_dir, pattern = "\\.xlsx$", full.names = TRUE)
if (length(future_files) == 0) stop("No future xlsx files found in ", future_dir)
cat("Future files:", length(future_files), "\n")
cat(paste(" ", basename(future_files)), sep = "\n")

future_raw <- lapply(future_files, readxl::read_excel) |>
  bind_rows() |>
  mutate(basin_name = standardize_basin_names(basin_name))

missing_vars <- setdiff(bundle$driver_vars, names(future_raw))
if (length(missing_vars) > 0) stop("Missing driver variables in future data: ", paste(missing_vars, collapse = ", "))

for (v in bundle$driver_vars) {
  future_raw[[paste0(v, "_sc")]] <- (future_raw[[v]] - bundle$sc_means[v]) / bundle$sc_sds[v]
}
future_raw$basin_id <- factor(future_raw$basin_id, levels = bundle$basin_levels)

cat("Future rows:", nrow(future_raw), "\n")
cat("Years:", min(future_raw$year), "-", max(future_raw$year), "\n")
cat("Scaled predictor NA count:", sum(is.na(future_raw[bundle$sc_suffix])), "\n")

cat("========== 3. Predict future CHLA ==========\n")
future_raw$chla_pred <- predict(
  bundle$models[["chla"]],
  newdata = future_raw,
  exclude = "s(basin_id)",
  newdata.guaranteed = TRUE
)
cat("CHLA prediction NA:", sum(is.na(future_raw$chla_pred)), "\n")

pred_out_cols <- c("basin_id", "basin_name", "year", "model", "scenario", "chla_pred")
safe_write_csv(future_raw[, pred_out_cols], file.path(out_dir, "future_predictions_raw_chla.csv"))

cat("========== 4. TGR regional ensemble ==========\n")
pred_df <- future_raw |>
  left_join(zone_df, by = "basin_name")
if (any(is.na(pred_df$zone))) {
  warning("Unmatched basin_name values in zone table: ",
          paste(unique(pred_df$basin_name[is.na(pred_df$zone)]), collapse = ", "),
          call. = FALSE)
}
pred_combined <- bind_rows(pred_df, pred_df |> mutate(zone = "TGR"))
ensemble <- summarise_ensemble(pred_combined, "zone")$ensemble
safe_write_csv(ensemble, file.path(out_dir, "future_ensemble_chla.csv"))
print(ensemble |> filter(year == 2025) |> select(zone, scenario, chla_mean))

cat("========== 5. TGR regional plot ==========\n")
zone_order <- c("TGR", "Lower TGR", "Middle TGR", "Upper TGR")
make_4panel(ensemble, "zone", zone_order, "Predicted CHLA", "future_chla", out_dir)

cat("========== 6. Tributary ensemble and plot ==========\n")
tributary_order <- c("PengxiRiver", "ShennongRiver", "XiangxiRiver", "YuLinRiver")
tributary_raw <- future_raw |>
  filter(basin_name %in% tributary_order)
missing_tributaries <- setdiff(tributary_order, unique(tributary_raw$basin_name))
if (length(missing_tributaries) > 0) {
  stop("Missing tributaries in future data: ", paste(missing_tributaries, collapse = ", "))
}
safe_write_csv(tributary_raw[, pred_out_cols], file.path(trib_out_dir, "tributary_future_predictions_raw_chla.csv"))
tributary_ensemble <- summarise_ensemble(tributary_raw, "basin_name")$ensemble
safe_write_csv(tributary_ensemble, file.path(trib_out_dir, "tributary_future_ensemble_chla.csv"))
print(tributary_ensemble |> filter(year == 2025) |> select(basin_name, scenario, chla_mean))
make_4panel(tributary_ensemble, "basin_name", tributary_order,
            "Predicted CHLA", "tributary_future_chla", trib_out_dir)

cat("\n========== Outputs ==========\n")
cat("Regional:\n")
cat(" ", file.path(out_dir, "future_predictions_raw_chla.csv"), "\n")
cat(" ", file.path(out_dir, "future_ensemble_chla.csv"), "\n")
cat(" ", file.path(out_dir, "future_chla.pdf/.png"), "\n")
cat("Tributaries:\n")
cat(" ", file.path(trib_out_dir, "tributary_future_predictions_raw_chla.csv"), "\n")
cat(" ", file.path(trib_out_dir, "tributary_future_ensemble_chla.csv"), "\n")
cat(" ", file.path(trib_out_dir, "tributary_future_chla.pdf/.png"), "\n")
