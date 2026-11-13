
import React, { useEffect, useState } from 'react'

import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Marker,
  Polyline,
  Tooltip,
  useMapEvents,
  useMap
} from 'react-leaflet'

import L from 'leaflet'

import {
  ArrowLeft,
  ArrowRight,
  Eye,
  Layers,
  RotateCcw,
  ShieldCheck,
  Sparkles
} from 'lucide-react'

import './OSVLive.css'


// --------------------------------------------------
// MAP HELPERS
// --------------------------------------------------

function makePin(color) {
  return L.divIcon({
    className: 'osv-pin',
    html: `<span style="background:${color}"></span>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  })
}


function position(point) {
  if (!point) return null
  if (!Number.isFinite(point.lat)) return null
  if (!Number.isFinite(point.lng)) return null

  return [point.lat, point.lng]
}


function distance(a, b) {
  if (!position(a) || !position(b)) {
    return null
  }

  const rad = value => value * Math.PI / 180

  const h =
    Math.sin(rad(b.lat - a.lat) / 2) ** 2 +
    Math.cos(rad(a.lat)) *
    Math.cos(rad(b.lat)) *
    Math.sin(rad(b.lng - a.lng) / 2) ** 2

  return Math.round(
    12742.0176 *
    Math.asin(Math.min(1, Math.sqrt(h)))
  )
}


function fmt(value) {
  if (!Number.isFinite(value)) {
    return 'Unknown'
  }

  return `${value.toLocaleString('en-US')} km`
}


async function api(path, method = 'GET', body) {
  const response = await fetch(path, {
    method,
    headers: body
      ? { 'Content-Type': 'application/json' }
      : undefined,
    body: body
      ? JSON.stringify(body)
      : undefined
  })

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(
      typeof data.detail === 'string'
        ? data.detail
        : `HTTP ${response.status}`
    )
  }

  return data
}


// --------------------------------------------------
// MAP CLICK
// --------------------------------------------------

function MapClick({ locked, onGuess }) {
  useMapEvents({
    click(event) {
      if (!locked) {
        onGuess({
          lat: event.latlng.lat,
          lng: event.latlng.lng
        })
      }
    }
  })

  return null
}


// --------------------------------------------------
// AUTO ZOOM MAP TO MARKERS
// --------------------------------------------------

function Fit({ points }) {
  const map = useMap()
  const key = JSON.stringify(points)

  useEffect(() => {
    const valid = points.filter(Boolean)

    if (valid.length === 1) {
      map.setView(valid[0], 3)
    }

    if (valid.length > 1) {
      map.fitBounds(valid, {
        padding: [40, 40],
        maxZoom: 5
      })
    }
  }, [key, map])

  return null
}


// --------------------------------------------------
// WORLD MAP
// --------------------------------------------------

function GuessMap({
  guess,
  onGuess,
  locked,
  result,
  tested,
  privacyResult
}) {
  const [borders, setBorders] = useState(null)

  // Country boundaries, like the main gallery.
  useEffect(() => {
    fetch(
      'https://cdn.jsdelivr.net/gh/johan/world.geo.json@master/countries.geo.json'
    )
      .then(response => {
        if (!response.ok) {
          throw new Error(
            'Failed to load country borders'
          )
        }

        return response.json()
      })
      .then(data => {
        setBorders(data)
      })
      .catch(error => {
        console.error(
          'Country borders:',
          error
        )
      })
  }, [])

  const userPoint = position(guess)

  const truePoint = locked
    ? position(result?.truth)
    : null

  const aiPoint = locked
    ? position(result?.model)
    : null

  const protectedPoint = locked && tested
    ? position(privacyResult?.model)
    : null

  const points = [
    userPoint,
    truePoint,
    aiPoint,
    protectedPoint
  ].filter(Boolean)

  return (
    <div className="osv-map-wrap">
      <MapContainer
        className="osv-map"
        center={[18, 0]}
        zoom={2}
        minZoom={2}
        maxZoom={8}
        scrollWheelZoom={true}
        worldCopyJump={true}
      >
        {/* Background map without CARTO API key */}
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          maxZoom={19}
        />

        {/* Country borders */}
        {borders && (
          <GeoJSON
            data={borders}
            interactive={false}
            style={{
              color: '#64748b',
              weight: 0.9,
              opacity: 0.65,
              fillOpacity: 0
            }}
          />
        )}

        <MapClick
          locked={locked}
          onGuess={onGuess}
        />

        <Fit points={points} />

        {/* User guess - Blue */}
        {userPoint && (
          <Marker
            position={userPoint}
            icon={makePin('#3b82f6')}
            draggable={!locked}
            eventHandlers={{
              dragend(event) {
                const p = event.target.getLatLng()

                onGuess({
                  lat: p.lat,
                  lng: p.lng
                })
              }
            }}
          >
            <Tooltip direction="top">
              Your guess
            </Tooltip>
          </Marker>
        )}

        {/* Actual GPS - Green */}
        {truePoint && (
          <Marker
            position={truePoint}
            icon={makePin('#16a34a')}
          >
            <Tooltip direction="top">
              True location
            </Tooltip>
          </Marker>
        )}

        {/* Original AI - Orange */}
        {aiPoint && (
          <Marker
            position={aiPoint}
            icon={makePin('#f97316')}
          >
            <Tooltip direction="top">
              Original AI prediction
            </Tooltip>
          </Marker>
        )}

        {/* Protected AI - Purple */}
        {protectedPoint && (
          <Marker
            position={protectedPoint}
            icon={makePin('#a855f7')}
          >
            <Tooltip direction="top">
              Protected AI prediction
            </Tooltip>
          </Marker>
        )}

        {/* Dashed line: User -> Truth */}
        {userPoint && truePoint && (
          <Polyline
            positions={[
              userPoint,
              truePoint
            ]}
            pathOptions={{
              color: '#3b82f6',
              weight: 3,
              opacity: 0.9,
              dashArray: '7 8'
            }}
          />
        )}

        {/* Dashed line: AI -> Truth */}
        {aiPoint && truePoint && (
          <Polyline
            positions={[
              aiPoint,
              truePoint
            ]}
            pathOptions={{
              color: '#f97316',
              weight: 3,
              opacity: 0.9,
              dashArray: '7 8'
            }}
          />
        )}

        {/* Dashed line: Original AI -> Protected AI */}
        {aiPoint && protectedPoint && (
          <Polyline
            positions={[
              aiPoint,
              protectedPoint
            ]}
            pathOptions={{
              color: '#a855f7',
              weight: 3,
              opacity: 0.9,
              dashArray: '7 8'
            }}
          />
        )}
      </MapContainer>

      {/* Map legend */}
      <div className="osv-legend">
        <span>
          <i
            style={{
              background: '#3b82f6'
            }}
          />
          You
        </span>

        {locked && (
          <>
            <span>
              <i
                style={{
                  background: '#f97316'
                }}
              />
              AI
            </span>

            {truePoint && (
              <span>
                <i
                  style={{
                    background: '#16a34a'
                  }}
                />
                Truth
              </span>
            )}

            {protectedPoint && (
              <span>
                <i
                  style={{
                    background: '#a855f7'
                  }}
                />
                Protected AI
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}


// --------------------------------------------------
// MAIN OSV LIVE COMPONENT
// --------------------------------------------------

export default function OSVLive() {
  const [photos, setPhotos] = useState([])
  const [loading, setLoading] = useState(true)

  const [selected, setSelected] = useState(null)

  const [guess, setGuess] = useState(null)
  const [result, setResult] = useState(null)

  const [heatmap, setHeatmap] = useState(null)
  const [clues, setClues] = useState(null)
  const [importance, setImportance] = useState(null)

  const [blur, setBlur] = useState(null)
  const [protectedResult, setProtectedResult] = useState(null)

  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  const [showHeat, setShowHeat] = useState(false)
  const [showClues, setShowClues] = useState(false)


  // Load OSV image gallery.
  useEffect(() => {
    api('/api/osv/images')
      .then(data => {
        setPhotos(data.photos || [])
      })
      .catch(err => {
        setError(err.message)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])


  // Open a selected image or return to gallery.
  function open(photo) {
    setSelected(photo)

    setGuess(null)
    setResult(null)

    setHeatmap(null)
    setClues(null)
    setImportance(null)

    setBlur(null)
    setProtectedResult(null)

    setShowHeat(false)
    setShowClues(false)

    setBusy('')
    setError('')
  }


  // Run one asynchronous request.
  async function run(name, action) {
    if (busy) {
      return
    }

    setError('')
    setBusy(name)

    try {
      await action()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }


  // Run GeoViT on original image.
  function reveal() {
    if (!guess || !selected) {
      return
    }

    run(
      'GeoViT is analyzing the original image…',
      async () => {
        const data = await api(
          `/api/osv/${selected.id}/predict`,
          'POST'
        )

        setResult(data)
      }
    )
  }


  // Calculate a live occlusion heatmap.
  function getHeatmap() {
    if (showHeat) {
      setShowHeat(false)
      return
    }

    if (heatmap) {
      setShowHeat(true)
      setShowClues(false)
      return
    }

    run(
      'Calculating heatmap (live occlusion)…',
      async () => {
        const data = await api(
          `/api/osv/${selected.id}/heatmap`,
          'POST'
        )

        setHeatmap(data)

        setShowHeat(true)
        setShowClues(false)
      }
    )
  }


  // Run live segmentation.
  function getClues() {
    if (showClues) {
      setShowClues(false)
      return
    }

    if (clues) {
      setShowClues(true)
      setShowHeat(false)
      return
    }

    run(
      'Running live segmentation…',
      async () => {
        const data = await api(
          `/api/osv/${selected.id}/clues`,
          'POST'
        )

        setClues(data)

        setShowClues(true)
        setShowHeat(false)
      }
    )
  }


  // Heatmap per clue class and the continent's clue ranking.
  function getImportance() {
    run(
      'Comparing the heatmap with the clues…',
      async () => {
        const data = await api(
          `/api/osv/${selected.id}/importance`,
          'POST'
        )

        setImportance(data)
      }
    )
  }


  // Run privacy model.
  function protect() {
    run(
      'Privacy model is creating the protected image…',
      async () => {
        const data = await api(
          `/api/osv/${selected.id}/privacy`,
          'POST'
        )

        setBlur(data)

        setProtectedResult(null)

        setShowHeat(false)
        setShowClues(false)
      }
    )
  }


  // Run GeoViT on protected image.
  function testProtected() {
    if (!blur) {
      return
    }

    run(
      'GeoViT is analyzing the protected image…',
      async () => {
        const data = await api(
          `/api/osv/${selected.id}/predict-protected`,
          'POST',
          {
            image_data: blur.image_data
          }
        )

        setProtectedResult(data)
      }
    )
  }


  const original = selected
    ? `/api/osv/${selected.id}/image`
    : null

  const trueDistance = distance(
    guess,
    result?.truth
  )

  const modelDistance = distance(
    result?.model,
    result?.truth
  )

  const protectedDistance = distance(
    protectedResult?.model,
    result?.truth
  )

  const moved = distance(
    result?.model,
    protectedResult?.model
  )


  return (
    <div className="osv-page">
      <header className="osv-header">
        <a
          href="/"
          className="osv-back"
        >
          <ArrowLeft size={18} />
          Main gallery
        </a>

        <strong>
          GeoGuess<span>AI</span>
          <small> / OSV LIVE</small>
        </strong>

        <span className="osv-live-tag">
          ON-DEMAND INFERENCE
        </span>
      </header>

      {!selected ? (
        <main className="osv-container">
          <div className="osv-intro">
            <span>
              OSV LIVE GALLERY
            </span>

            <h1>
              Every prediction happens <em>now.</em>
            </h1>

            <p>
              Choose any photo in
              {' '}
              <code>osv_image_galary</code>.
              Unlike the main gallery, the models run
              when you click — no precomputed predictions
              or masks.
            </p>
          </div>

          {error && (
            <p className="osv-error">
              {error}
            </p>
          )}

          {loading ? (
            <p>Loading image folder…</p>
          ) : photos.length === 0 ? (
            <div className="osv-empty">
              No photos yet. Add JPG/PNG/WebP files to
              <code>
                {' '}
                webapp/osv_image_galary/
                {' '}
              </code>
              and refresh.
            </div>
          ) : (
            <div className="osv-gallery">
              {photos.map((photo, index) => (
                <button
                  key={photo.id}
                  className="osv-card"
                  onClick={() => open(photo)}
                >
                  <img
                    src={photo.thumbnail}
                    alt="Unknown location"
                    loading="lazy"
                  />

                  <span>
                    OSV IMAGE
                    {' '}
                    {String(index + 1).padStart(2, '0')}

                    <ArrowRight size={17} />
                  </span>
                </button>
              ))}
            </div>
          )}
        </main>
      ) : (
        <main className="osv-container">
          <button
            className="osv-back osv-return"
            onClick={() => open(null)}
          >
            <ArrowLeft size={17} />
            OSV gallery
          </button>

          <div className="osv-columns">
            {/* LEFT: IMAGE & AI FEATURES */}
            <section className="osv-panel">
              <div className="osv-panel-head">
                <span>
                  LIVE VISUAL ANALYSIS
                </span>

                <h2>
                  {blur
                    ? 'Protected image'
                    : 'Mystery photo'}
                </h2>
              </div>

              <div className="osv-photo-box">
                <img
                  className="osv-photo"
                  src={blur?.image_data || original}
                  alt="OSV test"
                />

                {!blur && showHeat && heatmap && (
                  <img
                    className="osv-overlay"
                    src={heatmap.heatmap_data}
                    alt="Live occlusion heatmap"
                  />
                )}

                {!blur && showClues && clues && (
                  <img
                    className="osv-overlay"
                    src={clues.overlay_data}
                    alt="Live segmentation"
                  />
                )}
              </div>

              {/* HEATMAP & SEGMENTATION */}
              {result && !blur && (
                <div className="osv-actions">
                  <button
                    disabled={!!busy}
                    onClick={getHeatmap}
                  >
                    <Eye size={17} />

                    {showHeat
                      ? 'Hide heatmap'
                      : 'Calculate heatmap'}
                  </button>

                  <button
                    disabled={!!busy}
                    onClick={getClues}
                  >
                    <Layers size={17} />

                    {showClues
                      ? 'Hide clues'
                      : 'Run segmentation'}
                  </button>
                </div>
              )}

              {/* SEGMENTATION CLUE LIST */}
              {showClues && clues && (
                <div className="osv-clue-list">
                  {clues.clues
                    .filter(clue => clue.share_pct >= 1)
                    .sort((a, b) => b.share_pct - a.share_pct)
                    .map(clue => (
                    <div key={clue.name}>
                      <i
                        style={{
                          background: clue.color
                        }}
                      />

                      {clue.name}

                      <strong>
                        {clue.share_pct}%
                      </strong>
                    </div>
                  ))}
                </div>
              )}

              {/* WHAT THE MODEL RELIED ON */}
              {result && !blur && !importance && (
                <div className="osv-actions">
                  <button
                    disabled={!!busy}
                    onClick={getImportance}
                  >
                    <Layers size={17} />
                    What did the model rely on?
                  </button>
                </div>
              )}

              {importance && !blur && (
                <div>
                  <div className="subheading">What the model relied on</div>
                  {importance.clue_importance?.length ? (
                    <>
                      <p className="muted">
                        How much the model's guess depended on each part of the photo (1× = an average spot).
                      </p>
                      {importance.clue_importance.slice(0, 5).map(c => (
                        <div className="rank" key={c.name}>
                          <strong>{c.name}</strong>
                          <div className="bar">
                            <i style={{ width: `${Math.max(4, 100 * c.heat_ratio / importance.clue_importance[0].heat_ratio)}%` }} />
                          </div>
                          <b>{c.heat_ratio.toFixed(1)}×</b>
                        </div>
                      ))}
                    </>
                  ) : (
                    <p className="muted">The model was so sure that no single region changed its mind.</p>
                  )}

                  {importance.continent_clues && (
                    <>
                      <div className="subheading">What gives {importance.continent} away</div>
                      <p className="muted">
                        Hiding this clue makes the model this much less sure than hiding a random patch of
                        the same size ({importance.continent_clues.photos} street-view photos).
                      </p>
                      {importance.continent_clues.clues.filter(c => c.effect > 0).slice(0, 3).map(c => (
                        <div className="rank" key={c.name} style={{ opacity: c.significant ? 1 : 0.45 }}>
                          <strong>{c.name}</strong>
                          <div className="bar">
                            <i style={{ width: `${100 * c.effect / importance.continent_clues.clues[0].effect}%` }} />
                          </div>
                          <b>−{Math.round(c.effect * 100)} pts</b>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}

              {/* PROTECT IMAGE */}
              {result && !blur && (
                <button
                  className="osv-primary"
                  disabled={!!busy}
                  onClick={protect}
                >
                  <ShieldCheck size={18} />
                  Protect image (run privacy model)
                </button>
              )}

              {/* TEST PROTECTED IMAGE */}
              {blur && !protectedResult && (
                <>
                  <p className="osv-note">
                    The privacy network has generated a new
                    protected image. Test it with GeoViT.
                  </p>

                  <button
                    className="osv-primary"
                    disabled={!!busy}
                    onClick={testProtected}
                  >
                    <Sparkles size={18} />
                    Test protected image
                  </button>
                </>
              )}

              {/* RETURN TO ORIGINAL */}
              {blur && (
                <button
                  className="osv-text-btn"
                  onClick={() => {
                    setBlur(null)
                    setProtectedResult(null)
                  }}
                >
                  ← Show original image
                </button>
              )}

              {/* PROTECTED IMAGE RESULTS */}
              {protectedResult && (
                <div className="osv-results">
                  <h3>
                    PROTECTED IMAGE · AI TEST RESULTS
                  </h3>

                  <div>
                    <small>TRUE LOCATION</small>

                    <strong>
                      {result?.truth?.country || 'Unknown'}
                    </strong>
                  </div>

                  <div>
                    <small>
                      AI PREDICTED AFTER PROTECTION
                    </small>

                    <strong>
                      {protectedResult.model?.country ||
                        'Unknown'}
                    </strong>
                  </div>

                  <div>
                    <small>PROTECTED AI ERROR</small>

                    <strong>
                      {fmt(protectedDistance)}
                    </strong>
                  </div>

                  <div>
                    <small>AI GUESS MOVED</small>

                    <strong>
                      {fmt(moved)}
                    </strong>
                  </div>

                  <div>
                    <small>BLURRED AREA</small>

                    <strong>
                      {blur.blurred_area_pct}%
                    </strong>
                  </div>

                  <div>
                    <small>IMAGE QUALITY · SSIM</small>

                    <strong>
                      {blur.ssim ?? 'Unavailable'}
                    </strong>
                  </div>
                </div>
              )}
            </section>

            {/* RIGHT: MAP & RESULTS */}
            <section className="osv-panel">
              <div className="osv-panel-head">
                <span>
                  GEOLOCATION CHALLENGE
                </span>

                <h2>
                  {result
                    ? 'The big reveal'
                    : 'Where was it taken?'}
                </h2>
              </div>

              <p className="osv-note">
                {result
                  ? 'Your location versus the live model prediction.'
                  : 'Click the map to guess. The AI only starts when you lock your guess.'}
              </p>

              <GuessMap
                guess={guess}
                onGuess={setGuess}
                locked={!!result}
                result={result}
                tested={!!protectedResult}
                privacyResult={protectedResult}
              />

              {!result ? (
                <button
                  className="osv-primary"
                  disabled={!guess || !!busy}
                  onClick={reveal}
                >
                  Lock in my guess · Run GeoViT
                  <ArrowRight size={18} />
                </button>
              ) : (
                <>
                  {/* ORIGINAL MODEL RESULTS */}
                  <div className="osv-results">
                    <h3>
                      LIVE GEOLOCATION RESULTS
                    </h3>

                    <div>
                      <small>TRUE COUNTRY</small>

                      <strong>
                        {result.truth?.country ||
                          'No GPS label provided'}
                      </strong>
                    </div>

                    <div>
                      <small>AI PREDICTION</small>

                      <strong>
                        {result.model?.country || 'Unknown'}
                      </strong>
                    </div>

                    <div>
                      <small>YOUR DISTANCE</small>

                      <strong>
                        {fmt(trueDistance)}
                      </strong>
                    </div>

                    <div>
                      <small>AI DISTANCE</small>

                      <strong>
                        {fmt(modelDistance)}
                      </strong>
                    </div>
                  </div>

                  {/* TOP 3 */}
                  <h3 className="osv-subhead">
                    Top 3 countries
                  </h3>

                  {result.top3?.map((country, index) => (
                    <div
                      className="osv-rank"
                      key={country.country}
                    >
                      <span>
                        {index + 1}.
                        {' '}
                        {country.country}
                      </span>

                      <div className="osv-bar">
                        <i
                          style={{
                            width:
                              `${country.probability_pct}%`
                          }}
                        />
                      </div>

                      <strong>
                        {country.probability_pct.toFixed(1)}%
                      </strong>
                    </div>
                  ))}

                  <p className="osv-note">
                    GeoViT predicts countries, not precise
                    GPS coordinates. The AI pin is a
                    representative country point.
                  </p>
                </>
              )}
            </section>
          </div>

          {/* LOADING */}
          {busy && (
            <div className="osv-busy">
              <span className="osv-spinner" />
              {busy}
            </div>
          )}

          {/* ERROR */}
          {error && (
            <div className="osv-error">
              {error}
            </div>
          )}

          {/* NEXT IMAGE */}
          <button
            className="osv-next"
            onClick={() => open(null)}
          >
            <RotateCcw size={17} />
            Next OSV image
          </button>
        </main>
      )}
    </div>
  )
}
