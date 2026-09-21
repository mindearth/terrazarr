# Demo: WSF3D Italy

The World Settlement Footprint 3D layer over Italy, 178 335 × 200 599 float64 pixels of building
height, as the GeoZarr pyramid this pipeline writes (chunk 256, shard 4096, mean, nodata 0, ten
levels), served straight from the public bucket by a TiTiler with a GeoZarr reader. Every tile
you see is cut on request from the pyramid level that matches the zoom: no tile cache, no
pre-rendering.

<div id="map" style="height: 70vh; min-height: 420px; border-radius: 6px; margin: 1em 0;"></div>
<p id="status" style="font-size: 0.85em; opacity: 0.8;"></p>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
(function () {
  var titiler = "https://d3ig9jk3zb7o9j.cloudfront.net";
  var store = "s3://me-public-assets/terrazarr/demo/WSF3Dv3_Italy.zarr";
  var style = "&rescale=0,30&colormap_name=inferno";
  // level L of the pyramid has 2**L native pixels per pixel; the native 0.0000898° matches Web
  // Mercator zoom 14, so zoom z reads level 14 - z (clamped to the 10 levels of the store)
  function levelFor(z) { return Math.max(0, Math.min(9, 14 - z)); }
  var PyramidLayer = L.TileLayer.extend({
    getTileUrl: function (coords) {
      return titiler + "/geozarr/tiles/WebMercatorQuad/" + coords.z + "/" + coords.x + "/" + coords.y
        + ".png?url=" + encodeURIComponent(store) + "&variables=" + encodeURIComponent(levelFor(coords.z) + ":data") + style;
    }
  });
  var map = L.map("map").setView([41.9, 12.5], 6);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO", maxZoom: 19, opacity: 0.9
  }).addTo(map);
  new PyramidLayer("", {
    minZoom: 5, maxZoom: 18, maxNativeZoom: 14, opacity: 0.95,
    attribution: "WSF3D &copy; DLR, CC BY 4.0 &middot; tiles by TiTiler from the GeoZarr pyramid"
  }).addTo(map);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png", { maxZoom: 19, pane: "overlayPane" }).addTo(map);
  var status = document.getElementById("status");
  function show() { status.textContent = "zoom " + map.getZoom() + " reads level " + levelFor(map.getZoom()) + " of the pyramid; click the map to read the building height at a native pixel"; }
  map.on("zoomend", show); show();
  map.on("click", function (e) {
    status.textContent = "reading " + e.latlng.lat.toFixed(4) + ", " + e.latlng.lng.toFixed(4) + " …";
    fetch(titiler + "/geozarr/point/" + e.latlng.lng + "," + e.latlng.lat + "?url=" + encodeURIComponent(store) + "&variables=" + encodeURIComponent("0:data"))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var v = d.values && d.values[0];
        status.textContent = "building height at " + e.latlng.lat.toFixed(4) + ", " + e.latlng.lng.toFixed(4) + ": "
          + (v === null || v === undefined || v === 0 ? "no building" : v.toFixed(1) + " m") + " (level 0, one chunk read on demand)";
      })
      .catch(function () { status.textContent = "point query failed"; });
  });
})();
</script>

## What is on the map

- Colour is the mean building height of the pixel at the level shown, 0 to 30 m on the
  `inferno` colormap; nodata (no building) is transparent, so the dark basemap shows through.
- Zoom 5 to 14 are served from levels 9 down to 0 of the pyramid, 0.046° down to the native
  0.0000898°: the page names the level as `14 - zoom`, since level L has `2**L` native pixels
  per pixel and the native pixel matches zoom 14. Beyond 14 the browser upsamples the native tiles.
- A click reads the native pixel through the point endpoint.

## Behind the tiles

The store is `s3://me-public-assets/terrazarr/demo/WSF3Dv3_Italy.zarr`, written by:

```bash
terrazarr --input s3://me-public-assets/terrazarr/inputs/WSF3Dv3_Italy_full.zarr \
  --output WSF3Dv3_Italy.zarr \
  --chunk-size 256 --shard-size 4096 --sharding --method mean --nodata 0 \
  --workers 8 --threads-per-worker 1 --memory-limit 4GB --compressor zstd --clevel 3
```

six minutes on 8 processes, 2178 objects, 2.6 GB. The tile server is
[titiler](https://developmentseed.org/titiler/) with a GeoZarr reader (`/geozarr/`): a level is
named as `{level}:{variable}`, and the reader cuts the tile from that level's shard-aligned
chunks on S3, about 0.4 s per tile at every zoom. Asked for the root variable instead
(`/:data`) it reads the native array for any zoom and refuses tiles wider than a billion
pixels, so a client picks the level, as this page does. The same store answers:

- `GET {titiler}/geozarr/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url={store}&variables={14-z}:data&rescale=0,30&colormap_name=inferno`
- `GET {titiler}/geozarr/point/{lon},{lat}?url={store}&variables=0:data`
- `GET {titiler}/geozarr/info?url={store}&variables=9:data`

The data is the World Settlement Footprint 3D (DLR), CC BY 4.0; inputs and the pyramid are in
the public bucket described in [benchmarks](benchmarks.md#data).
