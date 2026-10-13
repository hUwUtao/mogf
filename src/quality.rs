use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub(super) enum RenderProfile {
    Preview,
    #[default]
    Final,
}

impl RenderProfile {
    pub(super) fn text_tolerance_scale(self) -> f64 {
        match self {
            Self::Preview => 4.0,
            Self::Final => 1.0,
        }
    }

    pub(super) fn shadow_resolution(self, authored: u32) -> u32 {
        match self {
            Self::Preview => authored.min(256),
            Self::Final => authored,
        }
    }

    pub(super) fn shadow_filter_radius(self, authored: f64) -> f64 {
        match self {
            Self::Preview => authored,
            Self::Final => authored,
        }
    }
}
