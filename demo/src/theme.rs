use eframe::egui::{
    self, Color32, CornerRadius, FontFamily, FontId, Margin, RichText, Stroke, TextStyle, Vec2,
};
use rtmpose::shared::colors;

const fn rgb(c: [u8; 3]) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

pub const INK_950: Color32 = Color32::from_rgb(0x0A, 0x0A, 0x0A);
pub const INK_900: Color32 = Color32::from_rgb(0x0D, 0x0D, 0x0D);
pub const INK_800: Color32 = Color32::from_rgb(0x11, 0x11, 0x11);
pub const INK_700: Color32 = Color32::from_rgb(0x14, 0x14, 0x14);
pub const INK_500: Color32 = Color32::from_rgb(0x1F, 0x1F, 0x1F);
pub const INK_400: Color32 = Color32::from_rgb(0x2A, 0x2A, 0x2A);
pub const INK_300: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
pub const INK_200: Color32 = Color32::from_rgb(0x8A, 0x8A, 0x8A);

pub const BACKGROUND: Color32 = INK_950;
pub const SURFACE: Color32 = INK_800;
pub const BORDER: Color32 = INK_500;
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xF5, 0xF5, 0xF5);
pub const TEXT_SECONDARY: Color32 = INK_200;
pub const TEXT_TERTIARY: Color32 = INK_300;

pub const RUST: Color32 = rgb(colors::RUST);
pub const TURQUOISE: Color32 = rgb(colors::TURQUOISE);

pub fn apply(ctx: &egui::Context) {
    install_text_styles(ctx);
    install_visuals(ctx);
}

fn install_text_styles(ctx: &egui::Context) {
    ctx.all_styles_mut(|style| {
        use TextStyle::*;
        style.text_styles = [
            (Heading, FontId::new(26.0, FontFamily::Proportional)),
            (
                Name("Label".into()),
                FontId::new(11.0, FontFamily::Proportional),
            ),
            (Body, FontId::new(14.0, FontFamily::Proportional)),
            (Button, FontId::new(13.0, FontFamily::Proportional)),
            (Monospace, FontId::new(12.0, FontFamily::Monospace)),
            (Small, FontId::new(11.0, FontFamily::Proportional)),
        ]
        .into();
    });
}

fn install_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(TEXT_PRIMARY);
    visuals.panel_fill = BACKGROUND;
    visuals.window_fill = SURFACE;
    visuals.extreme_bg_color = INK_900;
    visuals.faint_bg_color = INK_700;
    visuals.window_stroke = Stroke::new(1.0_f32, BORDER);
    visuals.window_corner_radius = CornerRadius::same(12);
    visuals.menu_corner_radius = CornerRadius::same(12);
    visuals.selection.bg_fill = RUST;
    visuals.selection.stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
    visuals.hyperlink_color = RUST;

    let widgets = &mut visuals.widgets;
    widgets.noninteractive.bg_fill = SURFACE;
    widgets.noninteractive.weak_bg_fill = SURFACE;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
    widgets.noninteractive.corner_radius = CornerRadius::same(12);

    widgets.inactive.bg_fill = INK_700;
    widgets.inactive.weak_bg_fill = INK_700;
    widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT_SECONDARY);
    widgets.inactive.corner_radius = CornerRadius::same(8);

    widgets.hovered.bg_fill = INK_400;
    widgets.hovered.weak_bg_fill = INK_400;
    widgets.hovered.bg_stroke = Stroke::new(1.0_f32, INK_200);
    widgets.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
    widgets.hovered.corner_radius = CornerRadius::same(8);

    widgets.active.bg_fill = RUST;
    widgets.active.weak_bg_fill = RUST;
    widgets.active.bg_stroke = Stroke::new(1.0_f32, RUST);
    widgets.active.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);
    widgets.active.corner_radius = CornerRadius::same(8);

    ctx.set_visuals(visuals);

    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = Vec2::new(12.0, 10.0);
        style.spacing.window_margin = Margin::same(16);
        style.spacing.menu_margin = Margin::same(12);
    });
}

pub fn heading(text: impl Into<String>) -> RichText {
    RichText::new(text)
        .color(TEXT_PRIMARY)
        .text_style(TextStyle::Heading)
}

pub fn eyebrow(text: impl Into<String>) -> RichText {
    RichText::new(text.into().to_uppercase())
        .color(TEXT_TERTIARY)
        .text_style(TextStyle::Name("Label".into()))
}

pub fn body(text: impl Into<String>) -> RichText {
    RichText::new(text).color(TEXT_SECONDARY)
}

pub fn card_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(CornerRadius::same(16))
        .inner_margin(Margin::same(20))
}

pub fn status_dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
}

pub fn panel_header(ui: &mut egui::Ui, eyebrow_text: &str, heading_text: &str, live: bool) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(eyebrow(eyebrow_text));
            ui.add_space(4.0);
            ui.label(heading(heading_text));
        });
        if live {
            ui.add_space(16.0);
            ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    status_dot(ui, TURQUOISE);
                    ui.label(body("LIVE"));
                });
            });
        }
    });
}
