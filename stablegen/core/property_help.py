"""Shared copy for settings exposed in both a dialog and the scene queue."""

ORBIT_HELP = {
    'engine': 'Renderer for each orbit frame. Workbench is a preview renderer; Eevee and Cycles can show material lighting. The choice affects render time and appearance.',
    'interpolation': 'Orbit speed over time: Linear keeps constant speed; Ease In/Out accelerates and decelerates.',
    'use_hdri': 'Use HDRI environment lighting for the orbit. The standalone export uses the scene environment; the queue can load its chosen HDRI before each export.',
    'hdri_rotation': 'Initial rotation offset of the HDRI around the vertical axis. Environment motion controls how it rotates during the orbit.',
    'env_mode': 'Environment motion relative to the camera: fixed, rotating with it, counter-rotating, or rotating while the camera stays fixed.',
    'use_denoiser': 'Enable Cycles denoising to reduce image noise. Applies to Cycles renders, not Workbench or Eevee.',
}
