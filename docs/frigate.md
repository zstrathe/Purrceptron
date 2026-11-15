# Frigate (optional remote viewing)

[Frigate](https://frigate.video/) is an open-source network video recorder. It was
used once, separately from this project, to watch the cats remotely while away. It
isn't part of the cat detection pipeline and isn't installed by
`scripts/setup_pi.sh`. This page records the setup in case it is wanted again.

## It can't share the camera with the pipeline

Both Frigate and `deploy/detect.py` read the USB camera (`/dev/video0`), and a V4L2
camera can only be opened by one program at a time. Stop the detection pipeline
(the dashboard's Stop button, or run the dashboard with `--no-camera`) before
starting Frigate's camera, and stop Frigate again before restarting detection.

## Install

Frigate runs in Docker. On the Pi:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER      # log out and back in afterwards
mkdir -p ~/frigate/config
```

`~/frigate/docker-compose.yaml`:

```yaml
services:
  frigate:
    container_name: frigate
    privileged: true
    restart: unless-stopped        # starts at boot; use "no" to start it by hand
    image: ghcr.io/blakeblackshear/frigate:stable
    shm_size: "256mb"
    devices:
      - /dev/video0:/dev/video0
    volumes:
      - /etc/localtime:/etc/localtime:ro
      - ./config:/config
      - /mnt/frigate-ssd:/media/frigate      # recordings and snapshots
      - type: tmpfs
        target: /tmp/cache
        tmpfs:
          size: 256000000
    ports:
      - "5000:5000"     # web UI, no login (LAN only)
      - "8971:8971"     # web UI with login
      - "8554:8554"     # RTSP restream
      - "8555:8555/tcp" # WebRTC
      - "8555:8555/udp"
```

Frigate writes its database (`config/frigate.db`) to the config folder, and its
recordings and snapshots to `/media/frigate`. Keep that on an external SSD, not the
Pi's SD card: an SD card wears out under continuous video writes. The SSD was
formatted exFAT and mounted with this `/etc/fstab` line (`nofail` lets the Pi boot
without it):

```
UUID=<ssd-uuid> /mnt/frigate-ssd exfat defaults,noatime,uid=1000,gid=1000,nofail 0 2
```

## Configuration

`~/frigate/config/config.yml` (Frigate 0.16). The camera was left disabled; enable
it in the web UI or set `enabled: true`.

```yaml
tls:
  enabled: false
auth:
  enabled: true
mqtt:
  enabled: false

cameras:
  feeder_cam:
    enabled: false
    ffmpeg:
      inputs:
        - path: /dev/video0
          input_args: [-f, v4l2, -input_format, mjpeg, -video_size, 1280x720, -framerate, '15']
          roles: [detect]
    detect:
      enabled: true
      width: 640
      height: 360
      fps: 5
    record:
      enabled: false
    snapshots:
      enabled: true
      bounding_box: false
      crop: true
      retain:
        default: 30        # days
    objects:
      track: [cat]

detectors:
  cpu1:
    type: cpu

version: 0.16-0
```

Detection runs on the CPU here (`detectors: cpu`), which is enough at 5 fps and
640x360. Start it with `cd ~/frigate && docker compose up -d`; the first login
password is printed in `docker logs frigate`.

## Viewing it away from home

A [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
exposed the web UI without opening router ports. `cloudflared` ran as a service on
the Pi with an ingress like this (`/etc/cloudflared/config.yml`):

```yaml
tunnel: <tunnel-id>
credentials-file: /etc/cloudflared/<tunnel-id>.json
ingress:
  - hostname: cats.example.com
    service: http://localhost:8971      # the UI port that requires a login
  - service: http_status:404
```

Point the tunnel at port 8971 (login required), never 5000 (no login). The tunnel
credentials file is a secret: keep it out of the repository and backups that leave
your network.
