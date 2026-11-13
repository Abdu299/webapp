
import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'

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
  ChevronRight,
  CircleHelp,
  Eye,
  Globe2,
  ImagePlus,
  Layers,
  MapPin,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Trophy,
  UploadCloud
} from 'lucide-react'

import OSVLive from './OSVLive'

import 'leaflet/dist/leaflet.css'
import './style.css'


// ==================================================
// MAP HELPERS
// ==================================================

function makePin(color) {
  return L.divIcon({
    className: 'custom-pin',
    html: `<div style="background:${color}"></div>`,
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

  const dLat = rad(b.lat - a.lat)
  const dLng = rad(b.lng - a.lng)

  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(rad(a.lat)) *
    Math.cos(rad(b.lat)) *
    Math.sin(dLng / 2) ** 2

  return Math.round(
    6371.0088 *
    2 *
    Math.asin(Math.min(1, Math.sqrt(h)))
  )
}


function fmt(value) {
  if (!Number.isFinite(value)) {
    return '—'
  }

  return `${value.toLocaleString('en-US')} km`
}


function assetUrl(id, name) {
  return `/api/photos/${encodeURIComponent(id)}/${name}`
}


async function jsonFetch(url, options) {
  const response = await fetch(url, options)

  const data = await response
    .json()
    .catch(() => ({}))

  if (!response.ok) {
    throw new Error(
      data.detail || `HTTP ${response.status}`
    )
  }

  return data
}


// ==================================================
// MAP CLICK HANDLER
// ==================================================

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


// ==================================================
// AUTO-ZOOM MAP
// ==================================================

function FitMarkers({ points }) {
  const map = useMap()

  const key = JSON.stringify(points)

  useEffect(() => {
    const valid = points.filter(Boolean)

    if (valid.length === 1) {
      map.setView(valid[0], 3)
    }

    if (valid.length > 1) {
      map.fitBounds(
        L.latLngBounds(valid).pad(0.32),
        {
          maxZoom: 5
        }
      )
    }
  }, [key, map])

  return null
}


// ==================================================
// WORLD MAP
// SAME STYLE AS OSV LIVE
// ==================================================

function WorldMap({
  guess,
  setGuess,
  locked,
  result,
  privacyTested
}) {
  const [borders, setBorders] = useState(null)

  // Load country borders
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

  // Visitor location
  const userPoint = position(guess)

  // True GPS
  const truePoint = locked
    ? position(result?.truth)
    : null

  // Original AI prediction
  const aiPoint = locked
    ? position(result?.model)
    : null

  // Privacy AI prediction
  const protectedPoint =
    locked && privacyTested
      ? position(result?.privacy?.model)
      : null

  // Used for automatic zoom
  const points = [
    userPoint,
    truePoint,
    aiPoint,
    protectedPoint
  ].filter(Boolean)

  return (
    <div className="map-shell">
      <MapContainer
        className="map"
        center={[18, 0]}
        zoom={2}
        minZoom={2}
        maxZoom={8}
        scrollWheelZoom={true}
        worldCopyJump={true}
      >
        {/* OpenStreetMap background */}
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
          onGuess={setGuess}
        />

        <FitMarkers points={points} />

        {/* BLUE: User's guess */}
        {userPoint && (
          <Marker
            position={userPoint}
            icon={makePin('#3b82f6')}
            draggable={!locked}
            eventHandlers={{
              dragend(event) {
                const point = event.target.getLatLng()

                setGuess({
                  lat: point.lat,
                  lng: point.lng
                })
              }
            }}
          >
            <Tooltip direction="top">
              Your guess
            </Tooltip>
          </Marker>
        )}

        {/* GREEN: Actual GPS */}
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

        {/* ORANGE: Original GeoViT prediction */}
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

        {/* PURPLE: Protected GeoViT prediction */}
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

        {/* BLUE LINE: User -> Truth */}
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

        {/* ORANGE LINE: AI -> Truth */}
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

        {/* PURPLE LINE: AI -> Protected AI */}
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
      <div className="map-key">
        <span>
          <i style={{ background: '#3b82f6' }} />
          You
        </span>

        {locked && (
          <>
            <span>
              <i style={{ background: '#f97316' }} />
              AI
            </span>

            {truePoint && (
              <span>
                <i style={{ background: '#16a34a' }} />
                Truth
              </span>
            )}

            {protectedPoint && (
              <span>
                <i style={{ background: '#a855f7' }} />
                Protected AI
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}


// ==================================================
// MAIN APPLICATION
// ==================================================

function App() {
  // OSV live page
  if (
    new URLSearchParams(
      window.location.search
    ).has('osv')
  ) {
    return <OSVLive />
  }

  return <MainGallery />
}


// ==================================================
// MAIN PRIVACY GALLERY
// ==================================================

function MainGallery() {
  const [gallery, setGallery] = useState([])
  const [loading, setLoading] = useState(true)

  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null)

  const [guess, setGuess] = useState(null)
  const [locked, setLocked] = useState(false)
  const [result, setResult] = useState(null)

  const [clues, setClues] = useState(false)
  const [heat, setHeat] = useState(false)

  const [privacy, setPrivacy] = useState(false)
  const [privacyTested, setPrivacyTested] = useState(false)

  const [uploading, setUploading] = useState(false)


  // Load precomputed Im2GPS gallery
  useEffect(() => {
    jsonFetch('/api/gallery')
      .then(data => {
        setGallery(data.photos || [])
      })
      .catch(error => {
        setError(error.message)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])


  // Return image URL
  function asset(name) {
    if (!selected) {
      return null
    }

    if (selected.uploaded) {
      if (name === 'original.jpg') {
        return selected.image_data
      }

      return null
    }

    return assetUrl(selected.id, name)
  }


  // Reset for the next visitor
  function restart() {
    setSelected(null)

    setGuess(null)
    setLocked(false)

    setResult(null)

    setClues(false)
    setHeat(false)

    setPrivacy(false)
    setPrivacyTested(false)

    setError('')
  }


  // Open image
  function choose(photo) {
    restart()
    setSelected(photo)
  }


  // Switch to precomputed privacy image
  function protectImage() {
    setPrivacy(true)
    setPrivacyTested(false)

    setClues(false)
    setHeat(false)
  }


  // Switch back to original
  function showOriginal() {
    setPrivacy(false)
    setPrivacyTested(false)
  }


  // Reveal saved original prediction
  async function reveal() {
    if (!guess || !selected) {
      return
    }

    setError('')

    try {
      let response

      if (selected.uploaded) {
        response = selected.result
      } else {
        response = await jsonFetch(
          `/api/photos/${selected.id}/reveal`,
          {
            method: 'POST'
          }
        )
      }

      setResult(response)
      setLocked(true)

    } catch (error) {
      setError(error.message)
    }
  }


  // Upload user photo
  async function upload(event) {
    const file = event.target.files?.[0]

    if (!file) {
      return
    }

    setUploading(true)
    setError('')

    const body = new FormData()
    body.append('file', file)

    try {
      const response = await jsonFetch(
        '/api/upload',
        {
          method: 'POST',
          body
        }
      )

      choose({
        ...response,
        uploaded: true
      })

    } catch (error) {
      setError(error.message)

    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }


  // Distances
  const userKm = distance(
    guess,
    result?.truth
  )

  const modelKm = distance(
    result?.model,
    result?.truth
  )

  const protectedKm = distance(
    result?.privacy?.model,
    result?.truth
  )

  const movedKm = distance(
    result?.model,
    result?.privacy?.model
  )


  // Determine the winner
  let winner = 'No ground-truth GPS available'

  if (
    userKm !== null &&
    modelKm !== null
  ) {
    if (userKm < modelKm) {
      winner = 'You beat the AI!'
    } else if (userKm > modelKm) {
      winner = 'AI wins this round'
    } else {
      winner = "It's a tie!"
    }
  }


  // Privacy availability
  const hasPrivacy =
    result?.privacy?.available &&
    result?.privacy?.model


  return (
    <div className="app">

      {/* HEADER */}
      <header className="top">
        <div className="brand">
          <span className="brand-icon">
            <Globe2 size={23} />
          </span>

          <div>
            <strong>
              GeoGuess<span>AI</span>
            </strong>

            <small>
              VISUAL GEOLOCATION · RESEARCH DEMO
            </small>
          </div>
        </div>

        <div className="top-right">
          <span className="pill">
            <span className="green-dot" />
            LIVE DEMO
          </span>

          <span className="top-sub">
            IN5310 · GROUP 2
          </span>
        </div>
      </header>


      {/* ================================================= */}
      {/* START PAGE / GALLERY */}
      {/* ================================================= */}

      {!selected ? (
        <main className="start">

          <div className="intro">
            <span className="eyebrow">
              CAN YOU OUTSMART THE MODEL?
            </span>

            <h1>
              Every photo tells
              <br />
              <em>a location.</em>
            </h1>

            <p>
              Study the scene. Drop your pin on the world map.
              Then discover what gave the location away —
              and how AI can hide it.
            </p>
          </div>


          {/* GALLERY HEADER */}
          <div className="section-title">
            <div>
              <h2>Choose your challenge</h2>

              <p>
                Pick a mystery photo to start your round.
              </p>
            </div>


            {/* UPLOAD + OSV BUTTONS */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                flexWrap: 'wrap'
              }}
            >
              <label className="upload">
                <UploadCloud size={17} />

                {uploading
                  ? 'Processing...'
                  : 'Upload your own'}

                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={upload}
                  disabled={uploading}
                />
              </label>

              <a
                href="/?osv=1"
                className="upload"
                style={{
                  textDecoration: 'none'
                }}
              >
                <Globe2 size={17} />
                OSV images
              </a>
            </div>
          </div>


          {error && (
            <div className="notice">
              {error}
            </div>
          )}


          {/* IMAGE GALLERY */}
          {loading ? (
            <p>Loading gallery…</p>

          ) : gallery.length === 0 ? (
            <div className="empty">
              <ImagePlus size={35} />

              <h3>No gallery images found</h3>

              <p>
                Generate the gallery from your GitHub
                repository and refresh the page.
              </p>
            </div>

          ) : (
            <div className="gallery">
              {gallery.map((photo, index) => (
                <button
                  key={photo.id}
                  className="photo-card"
                  onClick={() => choose(photo)}
                >
                  <img
                    src={photo.thumbnail}
                    alt="Mystery location"
                    loading="lazy"
                  />

                  <span className="photo-shade" />

                  <span className="photo-tag">
                    LOCATION
                    {' '}
                    {String(index + 1).padStart(2, '0')}
                  </span>

                  <span className="photo-caption">
                    <strong>
                      {photo.title}
                    </strong>

                    <ChevronRight size={19} />
                  </span>
                </button>
              ))}
            </div>
          )}


          <div className="footer-note">
            <ShieldCheck size={16} />

            Uploaded images are processed in memory.
            GPS metadata is removed.
          </div>

        </main>

      ) : (

        /* =============================================== */
        /* GAME PAGE */
        /* =============================================== */

        <main className="game">

          <div className="game-head">
            <button
              className="back"
              onClick={restart}
            >
              <ArrowLeft size={17} />
              Back to gallery
            </button>

            <span className="step">
              {locked
                ? 'ROUND REVEALED'
                : 'ROUND IN PROGRESS'}

              {' · '}
              {selected.title}
            </span>
          </div>


          <div className="game-grid">

            {/* ========================================= */}
            {/* LEFT PANEL: IMAGE */}
            {/* ========================================= */}

            <section className="panel scene-panel">

              <div className="panel-heading">
                <div>
                  <span className="eyebrow">
                    THE MYSTERY
                  </span>

                  <h2>
                    {privacy
                      ? 'Protected photo'
                      : 'Study the photo'}
                  </h2>
                </div>

                <span className="tiny-pill">
                  {privacy
                    ? 'PRIVACY PROTECTED'
                    : locked
                      ? 'REVEALED'
                      : 'LOCATION HIDDEN'}
                </span>
              </div>


              {/* IMAGE */}
              <div
                className={
                  privacy
                    ? 'photo-stage protected-stage'
                    : 'photo-stage'
                }
              >
                <img
                  key={
                    privacy
                      ? 'protected'
                      : 'original'
                  }
                  className="scene"
                  src={asset(
                    privacy
                      ? 'blurred.jpg'
                      : 'original.jpg'
                  )}
                  alt={
                    privacy
                      ? 'Protected photo'
                      : 'Original photo'
                  }
                />


                {/* SEGMENTATION OVERLAY */}
                {locked && !privacy && clues && (
                  <img
                    className="overlay"
                    src={asset('clues.png')}
                    alt="Visual clues"
                    onError={() => setClues(false)}
                  />
                )}


                {/* HEATMAP OVERLAY */}
                {locked && !privacy && heat && (
                  <img
                    className="overlay"
                    src={asset('heatmap.png')}
                    alt="AI heatmap"
                    onError={() => setHeat(false)}
                  />
                )}
              </div>


              {!locked && (
                <p className="hint">
                  <CircleHelp size={16} />

                  Look for road signs, architecture,
                  landscape and street markings.
                </p>
              )}


              {/* OVERLAY BUTTONS */}
              {locked && !privacy && (
                <>
                  <div className="toggles">
                    <button
                      className={
                        clues
                          ? 'toggle active'
                          : 'toggle'
                      }
                      disabled={
                        !result?.clues_available &&
                        !result?.clues?.length
                      }
                      onClick={() => {
                        setClues(!clues)
                        setHeat(false)
                      }}
                    >
                      <Layers size={16} />

                      Visual clues

                      <span>
                        {clues ? 'ON' : 'OFF'}
                      </span>
                    </button>

                    <button
                      className={
                        heat
                          ? 'toggle active'
                          : 'toggle'
                      }
                      disabled={
                        !result?.heatmap_available
                      }
                      onClick={() => {
                        setHeat(!heat)
                        setClues(false)
                      }}
                    >
                      <Eye size={16} />

                      AI heatmap

                      <span>
                        {heat ? 'ON' : 'OFF'}
                      </span>
                    </button>
                  </div>


                  {/* CLUE BREAKDOWN */}
                  <div className="subheading">
                    Visual clues
                  </div>

                  {result?.clues?.length ? (
                    <div className="clue-list">
                      {result.clues.map(
                        (clue, index) => (
                          <div
                            className="clue"
                            key={index}
                          >
                            <i
                              style={{
                                background:
                                  clue.color ||
                                  '#64748b'
                              }}
                            />

                            <span>
                              {clue.name}
                            </span>

                            <strong>
                              {clue.share_pct == null
                                ? '—'
                                : `${clue.share_pct}%`}
                            </strong>
                          </div>
                        )
                      )}
                    </div>
                  ) : (
                    <p className="muted">
                      Clue breakdown unavailable for
                      this photo.
                    </p>
                  )}
                </>
              )}


              {/* ======================================= */}
              {/* PRIVACY WORKFLOW */}
              {/* ======================================= */}

              {locked && hasPrivacy && (
                <div className="privacy-flow">

                  {/* STEP 1: PROTECT */}
                  {!privacy ? (
                    <button
                      className="primary privacy-action"
                      onClick={protectImage}
                    >
                      <ShieldCheck size={18} />

                      Protect image

                      <ArrowRight size={18} />
                    </button>

                  ) : !privacyTested ? (

                    /* STEP 2: TEST */
                    <>
                      <div className="privacy-info">
                        <ShieldCheck size={17} />

                        The protected image has replaced
                        the original. Test it to see what
                        the AI predicts.
                      </div>

                      <button
                        className="primary privacy-action"
                        onClick={() => {
                          setPrivacyTested(true)
                        }}
                      >
                        <Sparkles size={18} />

                        Test protected image

                        <ArrowRight size={18} />
                      </button>

                      <button
                        className="privacy-link"
                        onClick={showOriginal}
                      >
                        Show original image
                      </button>
                    </>

                  ) : (

                    /* STEP 3: PRIVACY RESULTS */
                    <div className="privacy-reveal">
                      <span className="eyebrow">
                        PROTECTED IMAGE · AI TEST RESULTS
                      </span>

                      <div className="privacy-answer">

                        <div>
                          <small>
                            TRUE LOCATION
                          </small>

                          <strong>
                            {result.truth?.country ||
                              'Unknown'}
                          </strong>
                        </div>

                        <div>
                          <small>
                            AI PREDICTED AFTER PROTECTION
                          </small>

                          <strong>
                            {result.privacy.model?.country ||
                              result.privacy.model?.label ||
                              'Unknown'}
                          </strong>
                        </div>
                      </div>


                      {/* PRIVACY STATISTICS */}
                      <div className="privacy-stats">

                        <div>
                          <small>
                            PROTECTED AI ERROR
                          </small>

                          <strong>
                            {fmt(protectedKm)}
                          </strong>
                        </div>

                        <div>
                          <small>
                            AI GUESS MOVED
                          </small>

                          <strong>
                            {fmt(movedKm)}
                          </strong>
                        </div>

                        <div>
                          <small>
                            IMAGE QUALITY ·
                            {' '}
                            {result.privacy.quality_metric ||
                              'SSIM'}
                          </small>

                          <strong>
                            {result.privacy.quality_score == null
                              ? '—'
                              : Number(
                                  result.privacy.quality_score
                                ).toFixed(3)}
                          </strong>
                        </div>
                      </div>

                      <p className="disclaimer">
                        GeoViT predicts countries, not
                        exact GPS coordinates. Distances
                        use representative locations for
                        predicted countries.
                      </p>

                      <button
                        className="privacy-link"
                        onClick={showOriginal}
                      >
                        ← Show original image
                      </button>
                    </div>
                  )}
                </div>
              )}
            </section>


            {/* ========================================= */}
            {/* RIGHT PANEL: MAP */}
            {/* ========================================= */}

            <section className="panel map-panel">

              <div className="panel-heading">
                <div>
                  <span className="eyebrow">
                    THE CHALLENGE
                  </span>

                  <h2>
                    {locked
                      ? 'The big reveal'
                      : 'Where was it taken?'}
                  </h2>
                </div>

                <MapPin
                  color="#64748b"
                  size={23}
                />
              </div>

              <p className="map-intro">
                {locked
                  ? 'Compare your location with the AI prediction.'
                  : 'Click anywhere on the map to place your guess. Drag your pin to refine it.'}
              </p>


              {/* NEW MAP */}
              <WorldMap
                guess={guess}
                setGuess={setGuess}
                locked={locked}
                result={result}
                privacyTested={privacyTested}
              />


              {/* LOCK GUESS BUTTON */}
              {!locked ? (
                <button
                  className="primary"
                  onClick={reveal}
                  disabled={!guess}
                >
                  Lock in my guess

                  <ArrowRight size={18} />
                </button>

              ) : (
                <>
                  {/* VERDICT */}
                  <div className="verdict">
                    <Trophy size={21} />

                    <div>
                      <strong>
                        {winner}
                      </strong>

                      {userKm !== null &&
                        modelKm !== null && (
                          <small>
                            {fmt(
                              Math.abs(
                                userKm - modelKm
                              )
                            )}

                            {' '}

                            {userKm < modelKm
                              ? 'closer than the AI'
                              : userKm > modelKm
                                ? 'closer for the AI'
                                : 'difference'}
                          </small>
                        )}
                    </div>
                  </div>


                  {/* DISTANCES */}
                  <div className="metrics">
                    <div>
                      <small>
                        YOUR DISTANCE
                      </small>

                      <strong>
                        {fmt(userKm)}
                      </strong>
                    </div>

                    <div>
                      <small>
                        AI DISTANCE
                      </small>

                      <strong>
                        {fmt(modelKm)}
                      </strong>
                    </div>
                  </div>


                  {/* COUNTRY LABELS */}
                  <div className="answer">
                    <div>
                      <small>
                        ACTUAL LOCATION
                      </small>

                      <strong>
                        {result.truth?.country ||
                          'Unspecified'}
                      </strong>
                    </div>

                    <div>
                      <small>
                        AI PREDICTION
                      </small>

                      <strong>
                        {result.model?.country ||
                          result.model?.label ||
                          'Unspecified'}
                      </strong>
                    </div>
                  </div>

                  <p className="disclaimer">
                    AI pins represent countries rather than
                    exact predicted GPS locations.
                  </p>
                </>
              )}
            </section>
          </div>


          {/* =========================================== */}
          {/* LOWER RESULTS */}
          {/* =========================================== */}

          {locked && (
            <div className="details-grid">

              {/* TOP 3 COUNTRIES */}
              <section className="panel result-panel">
                <div className="panel-heading">
                  <div>
                    <span className="eyebrow">
                      MODEL CONFIDENCE
                    </span>

                    <h2>
                      Top 3 countries
                    </h2>
                  </div>

                  <Sparkles size={20} />
                </div>

                {result.top3?.length ? (
                  result.top3
                    .slice(0, 3)
                    .map((country, index) => (
                      <div
                        className="rank"
                        key={index}
                      >
                        <span>
                          0{index + 1}
                        </span>

                        <strong>
                          {country.country}
                        </strong>

                        <div className="bar">
                          <i
                            style={{
                              width: `${
                                Math.max(
                                  0,
                                  Math.min(
                                    100,
                                    country.probability_pct ?? 0
                                  )
                                )
                              }%`
                            }}
                          />
                        </div>

                        <b>
                          {Number(
                            country.probability_pct ?? 0
                          ).toFixed(1)}%
                        </b>
                      </div>
                    ))
                ) : (
                  <p className="muted">
                    Country confidence data unavailable.
                  </p>
                )}
              </section>


              {/* PRIVACY INFORMATION */}
              <section className="panel result-panel">
                <div className="panel-heading">
                  <div>
                    <span className="eyebrow">
                      PRIVACY EXPERIMENT
                    </span>

                    <h2>
                      Can we hide the clues?
                    </h2>
                  </div>

                  <ShieldCheck size={23} />
                </div>

                <p className="map-intro">
                  {privacyTested
                    ? 'The protected prediction is shown below the image. The purple pin shows the protected AI prediction.'
                    : privacy
                      ? 'The protected image is displayed. Click Test protected image to reveal the new prediction.'
                      : 'Click Protect image below the original photo to see how privacy protection changes the AI prediction.'}
                </p>

                {!result.privacy?.available && (
                  <p className="muted">
                    No precomputed privacy result
                    for this photo.
                  </p>
                )}
              </section>
            </div>
          )}


          {/* NEXT VISITOR */}
          {locked && (
            <button
              className="next"
              onClick={restart}
            >
              <RotateCcw size={17} />

              Next photo / new visitor

              <ArrowRight size={17} />
            </button>
          )}


          {error && (
            <div className="notice">
              {error}
            </div>
          )}
        </main>
      )}


      {/* FOOTER */}
      <footer className="foot">
        GEOLOCATION · INTERPRETABILITY · PRIVACY

        <span>
          UNIVERSITY OF OSLO · IN5310
        </span>
      </footer>
    </div>
  )
}


// ==================================================
// RENDER
// ==================================================

createRoot(
  document.getElementById('root')
).render(<App />)
