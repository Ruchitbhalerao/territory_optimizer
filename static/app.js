// State
let map;
let data = {
    dealers: [],
    ftcs: [],
    relationships: [],
    changes: []
};

// Map Layers
let layerGroupFtcs;
let layerGroupDealers;
let layerGroupOldLinks;
let layerGroupNewLinks;
let layerGroupDistances;
let currentJobId = null;
let selectedId = null;
let selectedType = null;
let showDistances = false;
let measuring = false;
let measurePoints = [];
let measureLayer;

const API_BASE = '/api/v1';

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    setupEventListeners();
    fetchInitialData();
});

function initMap() {
    // Center map on India
    map = L.map('map').setView([20.5937, 78.9629], 5);

    // Standard OpenStreetMap tiles (darkened via CSS)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap'
    }).addTo(map);

    layerGroupFtcs = L.layerGroup().addTo(map);
    layerGroupDealers = L.layerGroup().addTo(map);
    layerGroupOldLinks = L.layerGroup().addTo(map);
    layerGroupNewLinks = L.layerGroup().addTo(map);
    layerGroupDistances = L.layerGroup().addTo(map);

    map.on('click', (e) => {
        if (measuring) {
            addMeasurePoint(e.latlng);
            return;
        }
        selectedId = null;
        selectedType = null;
        updateHighlighting();
        renderDistances();
    });
}

function setupEventListeners() {
    document.getElementById('btn-generate').addEventListener('click', generateData);
    document.getElementById('btn-optimize').addEventListener('click', runOptimization);
    const toggleRadius = document.getElementById('toggle-limit-radius');
    const wrapper = document.getElementById('radius-input-wrapper');
    if (toggleRadius && wrapper) {
        toggleRadius.addEventListener('change', () => {
            wrapper.style.display = toggleRadius.checked ? 'block' : 'none';
        });
    }
    
    // Layer toggles
    document.getElementById('layer-ftcs').addEventListener('change', (e) => toggleLayer(layerGroupFtcs, e.target.checked));
    document.getElementById('layer-dealers').addEventListener('change', (e) => toggleLayer(layerGroupDealers, e.target.checked));
    document.getElementById('layer-old-links').addEventListener('change', (e) => toggleLayer(layerGroupOldLinks, e.target.checked));
    document.getElementById('layer-new-links').addEventListener('change', (e) => toggleLayer(layerGroupNewLinks, e.target.checked));
    document.getElementById('toggle-distances').addEventListener('change', (e) => {
        showDistances = e.target.checked;
        renderDistances();
    });
    document.getElementById('toggle-measure').addEventListener('change', (e) => {
        if (e.target.checked) {
            startMeasure();
        } else {
            stopMeasure();
        }
    });

    // Stats panel collapse
    const statsToggle = document.getElementById('stats-toggle');
    const statsBody = document.getElementById('stats-body');
    const statsArrow = statsToggle?.querySelector('.collapse-arrow');
    if (statsToggle && statsBody) {
        statsToggle.addEventListener('click', () => {
            const hidden = statsBody.classList.toggle('hidden');
            if (statsArrow) statsArrow.classList.toggle('collapsed', hidden);
        });
    }
}

function toggleLayer(layer, show) {
    if (show) {
        map.addLayer(layer);
    } else {
        map.removeLayer(layer);
    }
}

async function fetchInitialData() {
    try {
        await Promise.all([
            fetchDealers(),
            fetchFtcs(),
            fetchRelationships()
        ]);
        renderMap();
        updateStats();
    } catch (e) {
        console.error("Failed to load initial data", e);
    }
}

async function fetchDealers() {
    const res = await fetch(`${API_BASE}/data/dealers`);
    if (res.ok) {
        data.dealers = await res.json();
    }
}

async function fetchFtcs() {
    const res = await fetch(`${API_BASE}/data/ftcs`);
    if (res.ok) {
        data.ftcs = await res.json();
    }
}

async function fetchRelationships() {
    const res = await fetch(`${API_BASE}/data/relationships`);
    if (res.ok) {
        data.relationships = await res.json();
    }
}

async function generateData() {
    const btn = document.getElementById('btn-generate');
    const statusEl = document.getElementById('gen-status');
    const dealers = document.getElementById('dealers-input').value;
    const ftcs = document.getElementById('ftcs-input').value;

    btn.disabled = true;
    btn.innerHTML = '<div class="loader"></div> Generating...';
    statusEl.innerHTML = '';

    try {
        const res = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dealers, ftcs })
        });
        
        const json = await res.json();
        if (res.ok) {
            statusEl.innerHTML = `<span class="text-success">Data generated successfully!</span>`;
            // clear old changes
            data.changes = [];
            layerGroupNewLinks.clearLayers();
            currentJobId = null;
            document.getElementById('stat-status').innerText = 'Pending';
            
            // fetch new data
            await fetchInitialData();
        } else {
            statusEl.innerHTML = `<span class="text-error">Error: ${json.message}</span>`;
        }
    } catch (e) {
        statusEl.innerHTML = `<span class="text-error">Error: ${e.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Generate Data';
    }
}

async function runOptimization() {
    const btn = document.getElementById('btn-optimize');
    const statusEl = document.getElementById('opt-status');
    const statStatus = document.getElementById('stat-status');

    btn.disabled = true;
    btn.innerHTML = '<div class="loader"></div> Optimizing...';
    statusEl.innerHTML = '';
    statStatus.innerText = 'Running...';

    try {
        const params = { "solver.time_limit_seconds": 30 };
        if (document.getElementById('toggle-minimize-disruption').checked) {
            params["optimization.lambda"] = 5.0;
        }
        if (document.getElementById('toggle-limit-radius').checked) {
            const radiusInput = document.getElementById('input-max-radius');
            params["constraints.max_travel_radius_km"] = parseFloat(radiusInput?.value) || 50;
        } else {
            params["constraints.max_travel_radius_km"] = 999;
        }
        const res = await fetch(`${API_BASE}/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ parameters: params })
        });
        
        const json = await res.json();
        if (res.ok) {
            statusEl.innerHTML = `<span class="text-success">Optimization complete! (Job: ${json.job_id})</span>`;
            statStatus.innerText = 'Optimal';
            currentJobId = json.job_id;
            
            // fetch new territories
            await fetchChanges(currentJobId);
        } else {
            statusEl.innerHTML = `<span class="text-error">Error: ${json.message}</span>`;
            statStatus.innerText = 'Failed';
        }
    } catch (e) {
        statusEl.innerHTML = `<span class="text-error">Error: ${e.message}</span>`;
        statStatus.innerText = 'Failed';
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Run Optimization';
    }
}

async function fetchChanges(jobId) {
    try {
        const res = await fetch(`${API_BASE}/solution/${jobId}/changes`);
        if (res.ok) {
            const json = await res.json();
            data.changes = json.changes || [];
            renderNewTerritories();
            updateStatsPanel();
        }
    } catch (e) {
        console.error("Failed to fetch changes", e);
    }
}

function updateStatsPanel() {
    const dealers = data.dealers;
    const ftcs = data.ftcs;
    const changes = data.changes;
    const relationships = data.relationships;

    // Build final assignment map
    const assignment = {};
    relationships.forEach(r => { assignment[r.dealer_id] = r.ftc_id; });
    changes.forEach(c => { assignment[c.dealer_id] = c.to_ftc; });

    const totalDealers = dealers.length;
    const totalFtcs = ftcs.length;

    // Retained / changed
    const changedSet = new Set(changes.map(c => c.dealer_id));
    const retained = totalDealers - changedSet.size;
    const retainedPct = totalDealers ? (retained / totalDealers * 100) : 0;
    document.getElementById('stat-retained').textContent = `${retained} / ${totalDealers} (${retainedPct.toFixed(1)}%)`;
    document.getElementById('stat-changed').textContent = `${changedSet.size} / ${totalDealers} (${(100 - retainedPct).toFixed(1)}%)`;

    // Unallocated dealers
    const unallocDealers = dealers.filter(d => !assignment[d.dealer_id]);
    document.getElementById('stat-unalloc-dealers').textContent = unallocDealers.length;

    // Per-FTC counts
    const ftcCounts = {};
    Object.values(assignment).forEach(fid => {
        ftcCounts[fid] = (ftcCounts[fid] || 0) + 1;
    });
    const activeFtcs = Object.keys(ftcCounts).length;
    const idleFtcs = totalFtcs - activeFtcs;
    document.getElementById('stat-active-ftcs').textContent = `${activeFtcs} / ${totalFtcs}`;
    document.getElementById('stat-idle-ftcs').textContent = idleFtcs;

    // Distribution stats
    const counts = Object.values(ftcCounts);
    if (counts.length) {
        const min = Math.min(...counts);
        const max = Math.max(...counts);
        const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
        const sorted = [...counts].sort((a, b) => a - b);
        const median = sorted.length % 2 === 0
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
            : sorted[Math.floor(sorted.length / 2)];
        document.getElementById('stat-min-dealers').textContent = min;
        document.getElementById('stat-max-dealers').textContent = max;
        document.getElementById('stat-mean-dealers').textContent = mean.toFixed(1);
        document.getElementById('stat-median-dealers').textContent = median.toFixed(1);
    }

    // Color-code warnings
    const elActive = document.getElementById('stat-active-ftcs');
    elActive.className = 'stat-row-value' + (activeFtcs > 0 ? ' good' : ' bad');
    const elIdle = document.getElementById('stat-idle-ftcs');
    elIdle.className = 'stat-row-value' + (idleFtcs > 0 ? ' warn' : ' good');
}

function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dlat = (lat2 - lat1) * Math.PI / 180;
    const dlon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dlat / 2) ** 2
            + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dlon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function getDealersForFtc(ftcId) {
    const assignments = {};
    data.relationships.forEach(r => { assignments[r.dealer_id] = r.ftc_id; });
    data.changes.forEach(c => { assignments[c.dealer_id] = c.to_ftc; });
    return Object.entries(assignments)
        .filter(([_, fid]) => fid === ftcId)
        .map(([did]) => data.dealers.find(d => d.dealer_id === did))
        .filter(Boolean);
}

function getFtcCoords(ftcId) {
    const dealers = getDealersForFtc(ftcId);
    if (dealers.length === 0) return null;
    const lat = dealers.reduce((s, d) => s + d.dealer_latitude, 0) / dealers.length;
    const lon = dealers.reduce((s, d) => s + d.dealer_longitude, 0) / dealers.length;
    return { lat, lon };
}

function renderDistances() {
    layerGroupDistances.clearLayers();
    if (!showDistances || !selectedId || selectedType !== 'ftc') return;

    const ftcId = selectedId;
    const dealers = getDealersForFtc(ftcId);
    const ftcCoords = getFtcCoords(ftcId);
    if (!ftcCoords || dealers.length === 0) return;

    const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const changedDealers = new Set(data.changes.map(c => c.dealer_id));

    // ---- FTC → Dealer ----
    // Place a clean pill label at the midpoint of each existing purple link.
    // We re-create the link here as a thin radial line so the label has a
    // visual anchor independent of the territory layer.
    const ftcLatLng = [ftcCoords.lat, ftcCoords.lon];
    dealers.forEach(d => {
        const dist = haversineKm(ftcCoords.lat, ftcCoords.lon, d.dealer_latitude, d.dealer_longitude);
        const dealerLatLng = [d.dealer_latitude, d.dealer_longitude];
        const lineColor = data.changes.length > 0 && !changedDealers.has(d.dealer_id) ? '#22c55e' : 'rgba(255, 255, 255, 0.15)';
        L.polyline([ftcLatLng, dealerLatLng], {
            color: lineColor,
            weight: 1
        }).addTo(layerGroupDistances);

        const pt = mid(ftcLatLng, dealerLatLng);
        L.marker(pt, {
            icon: L.divIcon({
                className: 'dist-pill',
                html: `${dist.toFixed(1)} <span class="dist-unit">km</span>`,
                iconSize: [0, 0],
                iconAnchor: [0, 0]
            })
        }).addTo(layerGroupDistances);
    });

    // ---- Dealer ↔ Dealer ----
    // Sort dealers by angle around the FTC centroid and draw the polygon
    // edges.  This forms a clean territory boundary instead of a messy all‑
    // pairs web.  If the territory has more than 10 dealers we skip the
    // polygon to keep the map uncluttered.
    const maxPoly = 10;
    if (dealers.length >= 3 && dealers.length <= maxPoly) {
        const sorted = dealers.map(d => ({
            lat: d.dealer_latitude,
            lng: d.dealer_longitude,
            angle: Math.atan2(d.dealer_latitude - ftcCoords.lat,
                              d.dealer_longitude - ftcCoords.lon)
        })).sort((a, b) => a.angle - b.angle);

        for (let i = 0; i < sorted.length; i++) {
            const j = (i + 1) % sorted.length;
            const a = sorted[i], b = sorted[j];
            const dist = haversineKm(a.lat, a.lng, b.lat, b.lng);
            L.polyline([[a.lat, a.lng], [b.lat, b.lng]], {
                color: 'rgba(251, 191, 36, 0.25)',
                weight: 1.5,
                dashArray: '3 5'
            }).addTo(layerGroupDistances);

            const pt = mid([a.lat, a.lng], [b.lat, b.lng]);
            L.marker(pt, {
                icon: L.divIcon({
                    className: 'dist-pill dist-pill--dim',
                    html: `${dist.toFixed(1)} <span class="dist-unit">km</span>`,
                    iconSize: [0, 0],
                    iconAnchor: [0, 0]
                })
            }).addTo(layerGroupDistances);
        }
    }
}

function initMeasureLayer() {
    if (!measureLayer) {
        measureLayer = L.layerGroup().addTo(map);
    }
}

function startMeasure() {
    measuring = true;
    measurePoints = [];
    initMeasureLayer();
    measureLayer.clearLayers();
    document.getElementById('measure-info').classList.add('visible');
    document.getElementById('measure-total').textContent = '0.0';
    map.getContainer().style.cursor = 'crosshair';
}

function stopMeasure() {
    measuring = false;
    measurePoints = [];
    if (measureLayer) measureLayer.clearLayers();
    document.getElementById('measure-info').classList.remove('visible');
    map.getContainer().style.cursor = '';
}

function addMeasurePoint(latlng) {
    measurePoints.push(latlng);
    measureLayer.clearLayers();

    // Draw dots
    measurePoints.forEach((p, i) => {
        L.circleMarker([p.lat, p.lng], {
            radius: 5,
            color: '#fbbf24',
            fillColor: '#fbbf24',
            fillOpacity: 0.9,
            weight: 2
        }).addTo(measureLayer);

        L.marker([p.lat, p.lng], {
            icon: L.divIcon({
                className: '',
                html: `<div style="color:#fbbf24;font-size:11px;font-weight:700;text-shadow:0 0 4px #000;margin:-18px 0 0 10px;">${i + 1}</div>`,
                iconSize: [0, 0],
                iconAnchor: [0, 0]
            })
        }).addTo(measureLayer);
    });

    // Draw lines between consecutive points
    let total = 0;
    for (let i = 0; i < measurePoints.length - 1; i++) {
        const a = measurePoints[i], b = measurePoints[i + 1];
        const seg = haversineKm(a.lat, a.lng, b.lat, b.lng);
        total += seg;

        L.polyline([[a.lat, a.lng], [b.lat, b.lng]], {
            color: '#fbbf24',
            weight: 2,
            opacity: 0.8
        }).addTo(measureLayer);

        // Segment label
        const mid = [(a.lat + b.lat) / 2, (a.lng + b.lng) / 2];
        L.marker(mid, {
            icon: L.divIcon({
                className: 'dist-pill',
                html: `${seg.toFixed(1)} <span class="dist-unit">km</span>`,
                iconSize: [0, 0],
                iconAnchor: [0, 0]
            })
        }).addTo(measureLayer);
    }

    document.getElementById('measure-total').textContent = total.toFixed(1);
}

function renderMap() {
    layerGroupFtcs.clearLayers();
    layerGroupDealers.clearLayers();
    layerGroupOldLinks.clearLayers();
    
    // Create maps for quick lookup
    const ftcMap = {};
    const dealerMap = {};

    // Calculate approximate FTC coordinates based on dealer clusters
    data.relationships.forEach(r => {
        const d = data.dealers.find(dealer => dealer.dealer_id === r.dealer_id);
        if (d) {
            if (!ftcMap[r.ftc_id]) {
                ftcMap[r.ftc_id] = { latSum: 0, lonSum: 0, count: 0 };
            }
            ftcMap[r.ftc_id].latSum += d.dealer_latitude;
            ftcMap[r.ftc_id].lonSum += d.dealer_longitude;
            ftcMap[r.ftc_id].count += 1;
        }
    });

    const ftcIcon = L.divIcon({
        className: 'custom-div-icon',
        html: `<div style="background-color:#10b981;width:12px;height:12px;border-radius:50%;border:2px solid #fff;"></div>`,
        iconSize: [12, 12],
        iconAnchor: [6, 6]
    });

    for (const [ftcId, info] of Object.entries(ftcMap)) {
        info.lat = info.latSum / info.count;
        info.lon = info.lonSum / info.count;
        const marker = L.marker([info.lat, info.lon], { icon: ftcIcon }).bindPopup(`FTC: ${ftcId}`).addTo(layerGroupFtcs);
        marker.on('click', () => handleMarkerClick(ftcId, 'ftc'));
    }

    // Render Dealers (Blue)
    const dealerIcon = L.divIcon({
        className: 'custom-div-icon',
        html: `<div style="background-color:#3b82f6;width:6px;height:6px;border-radius:50%;border:1px solid rgba(255,255,255,0.5);"></div>`,
        iconSize: [6, 6],
        iconAnchor: [3, 3]
    });

    const initialAssignments = {};
    data.relationships.forEach(r => { initialAssignments[r.dealer_id] = r.ftc_id; });

    data.dealers.forEach(d => {
        dealerMap[d.dealer_id] = d;
        const currentFtc = initialAssignments[d.dealer_id] || 'None';
        const popupContent = `<b>Dealer:</b> ${d.dealer_id}<br><b>Original FTC:</b> ${currentFtc}`;
        
        const marker = L.marker([d.dealer_latitude, d.dealer_longitude], { 
            icon: dealerIcon,
            dealerId: d.dealer_id
        }).bindPopup(popupContent).addTo(layerGroupDealers);
        marker.on('click', () => handleMarkerClick(d.dealer_id, 'dealer'));
    });

    // Render Old Links (Red)
    data.relationships.forEach(r => {
        const d = dealerMap[r.dealer_id];
        const f = ftcMap[r.ftc_id];
        if (d && f) {
            L.polyline([
                [d.dealer_latitude, d.dealer_longitude],
                [f.lat, f.lon]
            ], { 
                color: 'rgba(239, 68, 68, 0.2)', 
                weight: 1,
                dealerId: r.dealer_id,
                ftcId: r.ftc_id
            }).addTo(layerGroupOldLinks);
        }
    });
    
    updateHighlighting();
}

function renderNewTerritories() {
    layerGroupNewLinks.clearLayers();
    
    // We need ftcMap again
    const ftcMap = {};
    data.relationships.forEach(r => {
        const d = data.dealers.find(dealer => dealer.dealer_id === r.dealer_id);
        if (d) {
            if (!ftcMap[r.ftc_id]) {
                ftcMap[r.ftc_id] = { latSum: 0, lonSum: 0, count: 0 };
            }
            ftcMap[r.ftc_id].latSum += d.dealer_latitude;
            ftcMap[r.ftc_id].lonSum += d.dealer_longitude;
            ftcMap[r.ftc_id].count += 1;
        }
    });
    for (const [ftcId, info] of Object.entries(ftcMap)) {
        info.lat = info.latSum / info.count;
        info.lon = info.lonSum / info.count;
    }

    const dealerMap = {};
    data.dealers.forEach(d => { dealerMap[d.dealer_id] = d; });

    // The changes array has {dealer_id, from_ftc, to_ftc}
    const newAssignments = {};
    data.relationships.forEach(r => {
        newAssignments[r.dealer_id] = r.ftc_id;
    });
    data.changes.forEach(c => {
        newAssignments[c.dealer_id] = c.to_ftc;
    });

    Object.keys(newAssignments).forEach(dealerId => {
        const ftcId = newAssignments[dealerId];
        const d = dealerMap[dealerId];
        const f = ftcMap[ftcId];
        if (d && f) {
            L.polyline([
                [d.dealer_latitude, d.dealer_longitude],
                [f.lat, f.lon]
            ], { 
                color: 'rgba(168, 85, 247, 0.4)', 
                weight: 2,
                dealerId: dealerId,
                ftcId: ftcId
            }).addTo(layerGroupNewLinks); // Purple
        }
    });

    // Update dealer popups with new assignment info + marker colors
    const initialAssignments = {};
    data.relationships.forEach(r => { initialAssignments[r.dealer_id] = r.ftc_id; });
    const changedDealers = new Set(data.changes.map(c => c.dealer_id));

    layerGroupDealers.eachLayer(layer => {
        const dId = layer.options.dealerId;
        const currentFtc = initialAssignments[dId] || 'None';
        const newFtc = newAssignments[dId] || currentFtc;
        
        let popupContent = `<b>Dealer:</b> ${dId}<br><b>Original FTC:</b> ${currentFtc}`;
        if (data.changes.length > 0) {
            popupContent += `<br><b>Optimized FTC:</b> ${newFtc}`;
            if (currentFtc !== newFtc) {
                popupContent += ` <span style="color:#10b981;">(Changed)</span>`;
            }
        }
        layer.setPopupContent(popupContent);

        // Color: unchanged = orange, changed = blue
        const isUnchanged = data.changes.length > 0 && !changedDealers.has(dId);
        const color = isUnchanged ? '#f97316' : '#3b82f6';
        layer.setIcon(L.divIcon({
            className: 'custom-div-icon',
            html: `<div style="background-color:${color};width:6px;height:6px;border-radius:50%;border:1px solid rgba(255,255,255,0.5);"></div>`,
            iconSize: [6, 6],
            iconAnchor: [3, 3]
        }));
    });
    
    updateHighlighting();
    renderDistances();
}

function handleMarkerClick(id, type) {
    if (selectedId === id) {
        selectedId = null;
        selectedType = null;
    } else {
        selectedId = id;
        selectedType = type;
    }
    updateHighlighting();
    renderDistances();
}

function updateHighlighting() {
    const isSelected = selectedId !== null;

    layerGroupOldLinks.eachLayer(layer => {
        const dId = layer.options.dealerId;
        const fId = layer.options.ftcId;
        
        if (!isSelected) {
            layer.setStyle({ color: 'rgba(239, 68, 68, 0.2)', weight: 1 });
        } else if ((selectedType === 'dealer' && dId === selectedId) || 
                   (selectedType === 'ftc' && fId === selectedId)) {
            layer.setStyle({ color: 'rgba(239, 68, 68, 0.9)', weight: 3 });
            layer.bringToFront();
        } else {
            layer.setStyle({ color: 'rgba(239, 68, 68, 0.02)', weight: 1 });
        }
    });

    layerGroupNewLinks.eachLayer(layer => {
        const dId = layer.options.dealerId;
        const fId = layer.options.ftcId;
        
        if (!isSelected) {
            layer.setStyle({ color: 'rgba(168, 85, 247, 0.4)', weight: 2 });
        } else if ((selectedType === 'dealer' && dId === selectedId) || 
                   (selectedType === 'ftc' && fId === selectedId)) {
            layer.setStyle({ color: 'rgba(168, 85, 247, 1.0)', weight: 4 });
            layer.bringToFront();
        } else {
            layer.setStyle({ color: 'rgba(168, 85, 247, 0.02)', weight: 1 });
        }
    });
}

function updateStats() {
    document.getElementById('stat-dealers').innerText = data.dealers.length > 0 ? `${data.dealers.length} (Sampled)` : '0';
    document.getElementById('stat-ftcs').innerText = data.ftcs.length;
}
