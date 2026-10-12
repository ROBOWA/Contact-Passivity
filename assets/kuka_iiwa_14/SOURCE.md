# Vendored model: KUKA LBR iiwa 14 (MuJoCo Menagerie)

| Field | Value |
| --- | --- |
| Upstream repository | https://github.com/google-deepmind/mujoco_menagerie |
| Upstream directory | `kuka_iiwa_14/` |
| Pinned commit | `0059d4335f8156206f63a35662313385f7ad6d74` (2026-10-07) |
| Model license | BSD-3-Clause (see `LICENSE`, inherited from Drake) |
| Repository license | The Menagerie root `LICENSE` is an aggregate of per-model licenses; its `kuka_iiwa_14/` section is identical to `LICENSE` here |

Files copied verbatim from the pinned commit: `iiwa14.xml`, `scene.xml`,
`assets/*.obj`, `LICENSE`, `README.md`, `CHANGELOG.md`. The upstream preview
image `iiwa_14.png` (2 MB) was not copied. `UPSTREAM_BLOBS.txt` lists the git
blob hashes of every upstream file at the pinned commit so the copy can be
verified.

The torque-actuated robot used by this project lives one directory up in
`assets/iiwa14_torque.xml`; it is derived from `iiwa14.xml` and documents its
own changes. Nothing in this directory is modified.

## Re-vendoring at the pinned commit

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
cd /tmp/menagerie && git checkout 0059d4335f8156206f63a35662313385f7ad6d74 && git sparse-checkout set kuka_iiwa_14
cp -R kuka_iiwa_14/assets kuka_iiwa_14/{iiwa14.xml,scene.xml,LICENSE,README.md,CHANGELOG.md} <repo>/assets/kuka_iiwa_14/
```
