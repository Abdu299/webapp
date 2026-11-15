
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
import { ISO2_TO_ISO3 } from './countryIso3'

import 'leaflet/dist/leaflet.css'
import './style.css'



// GeoJSON country identifiers use ISO-3 while our labels use ISO-2.
function isTrueCountryPolygon(feature, countryCode) {
  if (!countryCode) return false

  const code = String(countryCode).trim().toUpperCase()
  const iso3 = ISO2_TO_ISO3[code] || code
  const props = feature?.properties || {}
  const identifiers = [
    feature?.id,
    props.ISO_A3,
    props.iso_a3,
    props.ADM0_A3,
    props.adm0_a3,
    props.ISO3166_1_Alpha_3,
    props.ISO_A2,
    props.iso_a2,
    props.ISO3166_1_Alpha_2
  ]

  return identifiers.some(value => {
    if (value == null) return false
    const actual = String(value).trim().toUpperCase()
    return actual === iso3 || actual === code
  })
}

function countryMapStyle(feature, truthCode) {
  if (isTrueCountryPolygon(feature, truthCode)) {
    return {
      color: '#15803d',
      weight: 2.5,
      opacity: 1,
      fillColor: '#22c55e',
      fillOpacity: 0.43
    }
  }

  return {
    color: '#64748b',
    weight: 0.9,
    opacity: 0.65,
    fillOpacity: 0
  }
}

// ==================================================
// MAP HELPERS
// ==================================================


// Highlight only the ground-truth country and matching predictions.
function isCorrectCountry(candidate, truth) {
  if (!candidate || !truth) return false
  return String(candidate).trim().toUpperCase() === String(truth).trim().toUpperCase()
}

const correctCountryStyle = { color: '#16a34a' }

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
            key={locked ? (result?.truth?.country || 'revealed') : 'hidden'}
            data={borders}
            interactive={false}
            style={feature => countryMapStyle(
              feature,
              locked ? result?.truth?.country : null
            )}
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
  const [reliance, setReliance] = useState(false)

  const [privacy, setPrivacy] = useState(false)
  const [privacyTested, setPrivacyTested] = useState(false)

  const [uploading, setUploading] = useState(false)

  // Uploaded photos have nothing saved, so their clues, heatmap and blur
  // are worked out live and kept here for the rest of the round
  const [live, setLive] = useState({})
  const [working, setWorking] = useState('')


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
      return {
        'original.jpg': selected.image_data,
        'blurred.jpg': live.privacy?.image_data,
        'clues.png': live.clues?.overlay_data,
        'heatmap.png': live.heatmap?.heatmap_data
      }[name] || null
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
    setReliance(false)

    setPrivacy(false)
    setPrivacyTested(false)

    setLive({})
    setWorking('')

    setError('')
  }


  // Open image
  function choose(photo) {
    restart()
    setSelected(photo)
  }


  // Runs one step live for an uploaded photo, the first time only.
  // Uses the same endpoints as the street view page.
  async function runLive(key, step, message, body) {
    if (live[key]) {
      return live[key]
    }

    setWorking(message)
    setError('')

    try {
      const data = await jsonFetch(
        `/api/osv/${encodeURIComponent(selected.id)}/${step}`,
        body
          ? {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(body)
            }
          : { method: 'POST' }
      )

      setLive(current => ({ ...current, [key]: data }))
      return data

    } catch (error) {
      setError(error.message)
      return null

    } finally {
      setWorking('')
    }
  }


  async function toggleClues() {
    if (
      !clues &&
      selected?.uploaded &&
      !(await runLive('clues', 'clues', 'Finding the clues in the photo…'))
    ) {
      return
    }

    setClues(!clues)
    setHeat(false)
  }


  async function toggleHeat() {
    if (
      !heat &&
      selected?.uploaded &&
      !(await runLive('heatmap', 'heatmap', 'Working out the heatmap, this takes about 20 seconds…'))
    ) {
      return
    }

    setHeat(!heat)
    setClues(false)
  }


  async function toggleReliance() {
    if (
      !reliance &&
      selected?.uploaded &&
      !(await runLive('importance', 'importance', 'Comparing the heatmap with the clues, this can take about 20 seconds…'))
    ) {
      return
    }

    setReliance(!reliance)
  }


  // Gallery photos show the saved blurred version,
  // uploads are blurred live by the privacy model
  async function protectImage() {
    if (
      selected?.uploaded &&
      !(await runLive('privacy', 'privacy', 'Blurring the photo…'))
    ) {
      return
    }

    setPrivacy(true)
    setPrivacyTested(false)

    setClues(false)
    setHeat(false)
  }


  async function testProtected() {
    if (selected?.uploaded) {
      const blurred = live.privacy
      const guessed = await runLive(
        'protected',
        'predict-protected',
        'The model is guessing on the blurred photo…',
        { image_data: blurred.image_data }
      )

      if (!guessed) {
        return
      }

      setResult(current => ({
        ...current,
        privacy: {
          available: true,
          model: guessed.model,
          new_top3: guessed.top3,
          quality_metric: 'SSIM',
          quality_score: blurred.ssim
        }
      }))
    }

    setPrivacyTested(true)
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
  let winner = 'The true location is unknown'

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
  const hasPrivacy = selected?.uploaded || (
    result?.privacy?.available &&
    result?.privacy?.model
  )

  // Uploads read their clues and reliance from the live results
  const clueList = selected?.uploaded
    ? (live.clues?.clues || [])
        .filter(clue => clue.share_pct >= 1)
        .sort((a, b) => b.share_pct - a.share_pct)
    : result?.clues

  const relied = selected?.uploaded ? live.importance : result


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
              CAN YOU BEAT THE MODEL?
            </span>

            <h1>
              Can you tell
              <br />
              <em>where this is?</em>
            </h1>

            <p>
              Look at a photo and guess where it was taken. Then
              see what gave the location away, and whether
              blurring part of the photo can hide it.
            </p>
          </div>


          {/* GALLERY HEADER */}
          <div className="section-title">
            <div>
              <h2>Pick a photo</h2>

              <p>
                Each photo is a new round.
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
                Street view photos
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
                Build the gallery with build_from_repo.py,
                then refresh the page.
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

            Uploaded photos are never saved, and their
            GPS data is removed.
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
                      : 'Look at the photo'}
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
                        !!working || (
                          !selected?.uploaded &&
                          !result?.clues_available &&
                          !result?.clues?.length
                        )
                      }
                      onClick={toggleClues}
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
                        !!working || (
                          !selected?.uploaded &&
                          !result?.heatmap_available
                        )
                      }
                      onClick={toggleHeat}
                    >
                      <Eye size={16} />

                      AI heatmap

                      <span>
                        {heat ? 'ON' : 'OFF'}
                      </span>
                    </button>
                  </div>

                  {working && (
                    <p className="muted">{working}</p>
                  )}


                  {/* CLUE BREAKDOWN */}
                  {clues && (
                  <>
                  <div className="subheading">
                    Visual clues
                  </div>

                  {clueList?.length ? (
                    <div className="clue-list">
                      {clueList.map(
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
                      No clue breakdown for this photo.
                    </p>
                  )}
                  </>
                  )}

                  {/* WHAT THE MODEL RELIED ON */}
                  <div className="toggles">
                    <button
                      className={
                        reliance
                          ? 'toggle active'
                          : 'toggle'
                      }
                      disabled={
                        !!working || (
                          !selected?.uploaded &&
                          !result?.clues_available
                        )
                      }
                      onClick={toggleReliance}
                    >
                      <Sparkles size={16} />

                      What did the model rely on?

                      <span>
                        {reliance ? 'ON' : 'OFF'}
                      </span>
                    </button>
                  </div>

                  {reliance && (
                  <>
                  <div className="subheading">What the model relied on</div>
                  {relied?.clue_importance?.length ? (
                    <>
                      <p className="muted">
                        How much each part of the photo mattered for the guess. 1× means an average spot.
                      </p>
                      {relied.clue_importance.slice(0, 5).map(c => (
                        <div className="rank" key={c.name}>
                          <strong>{c.name}</strong>
                          <div className="bar">
                            <i style={{ width: `${Math.max(4, 100 * c.heat_ratio / relied.clue_importance[0].heat_ratio)}%` }} />
                          </div>
                          <b>{c.heat_ratio.toFixed(1)}×</b>
                        </div>
                      ))}
                    </>
                  ) : (
                    <p className="muted">The model was so sure that hiding any one part of the photo didn't change its guess.</p>
                  )}

                  {/* WHAT GIVES THE CONTINENT AWAY */}
                  {relied?.continent_clues && (
                    <>
                      <div className="subheading">What gives {relied.continent} away</div>
                      <p className="muted">
                        How much less sure the model gets when this clue is hidden, compared with hiding a
                        random patch of the same size. Based on {relied.continent_clues.photos} street view
                        photos from {relied.continent}.
                      </p>
                      {relied.continent_clues.clues.filter(c => c.effect > 0).slice(0, 3).map(c => (
                        <div className="rank" key={c.name} style={{ opacity: c.significant ? 1 : 0.45 }}>
                          <strong>{c.name}</strong>
                          <div className="bar">
                            <i style={{ width: `${100 * c.effect / relied.continent_clues.clues[0].effect}%` }} />
                          </div>
                          <b>{Math.round(c.effect * 100)} pts</b>
                        </div>
                      ))}
                    </>
                  )}
                  </>
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
                      disabled={!!working}
                      onClick={protectImage}
                    >
                      <ShieldCheck size={18} />

                      Protect the photo

                      <ArrowRight size={18} />
                    </button>

                  ) : !privacyTested ? (

                    /* STEP 2: TEST */
                    <>
                      <div className="privacy-info">
                        <ShieldCheck size={17} />

                        This is the protected photo. Test
                        it to see what the model guesses
                        now.
                      </div>

                      <button
                        className="primary privacy-action"
                        disabled={!!working}
                        onClick={testProtected}
                      >
                        <Sparkles size={18} />

                        Test the protected photo

                        <ArrowRight size={18} />
                      </button>

                      <button
                        className="privacy-link"
                        onClick={showOriginal}
                      >
                        Show the original
                      </button>
                    </>

                  ) : (

                    /* STEP 3: PRIVACY RESULTS */
                    <div className="privacy-reveal">
                      <span className="eyebrow">
                        THE PROTECTED PHOTO
                      </span>

                      <div className="privacy-answer">

                        <div>
                          <small>
                            TRUE LOCATION
                          </small>

                          <strong style={result.truth?.country ? correctCountryStyle : undefined}>
                            {result.truth?.country ||
                              'Unknown'}
                          </strong>
                        </div>

                        <div>
                          <small>
                            AI GUESS AFTER BLURRING
                          </small>

                          <strong style={isCorrectCountry(result.privacy.model?.country || result.privacy.model?.label, result.truth?.country) ? correctCountryStyle : undefined}>
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
                            AI ERROR AFTER BLURRING
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
                        The model guesses a country, not
                        an exact spot, so distances are
                        measured from the middle of the
                        country it guessed.
                      </p>

                      <button
                        className="privacy-link"
                        onClick={showOriginal}
                      >
                        Show the original
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
                  ? "Here is your guess next to the model's."
                  : 'Click the map to place your guess. You can drag the pin to move it.'}
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

                      <strong style={result.truth?.country ? correctCountryStyle : undefined}>
                        {result.truth?.country ||
                          'Unknown'}
                      </strong>
                    </div>

                    <div>
                      <small>
                        AI PREDICTION
                      </small>

                      <strong style={isCorrectCountry(result.model?.country || result.model?.label, result.truth?.country) ? correctCountryStyle : undefined}>
                        {result.model?.country ||
                          result.model?.label ||
                          'Unknown'}
                      </strong>
                    </div>
                  </div>

                  <p className="disclaimer">
                    The AI pin marks the middle of the
                    country it guessed.
                  </p>

                  {/* TOP 3 COUNTRIES */}
                  <div className="subheading">
                    The model's top 3 countries
                  </div>

                  {result.top3?.length ? (
                    result.top3
                      .slice(0, 3)
                      .map((country, index) => (
                        <div
                          className="rank"
                          key={index}
                          style={isCorrectCountry(country.country, result.truth?.country) ? { background: '#f0fdf4', borderRadius: 9, paddingLeft: 10, paddingRight: 10 } : undefined}
                        >
                          <span>
                            0{index + 1}
                          </span>

                          <strong style={isCorrectCountry(country.country, result.truth?.country) ? correctCountryStyle : undefined}>
                            {country.country}
                          </strong>

                          <div className="bar">
                            <i
                              style={{
                                background: isCorrectCountry(country.country, result.truth?.country) ? '#16a34a' : undefined,
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
                      No country scores for this photo.
                    </p>
                  )}
                </>
              )}
            </section>
          </div>


          {/* =========================================== */}
          {/* LOWER RESULTS */}
          {/* =========================================== */}

          {locked && (
            <div className="privacy-panel">

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
                    ? "The model's new guess is under the photo, and the purple pin shows it on the map."
                    : privacy
                      ? "The protected photo is showing. Click Test the protected photo to see the model's new guess."
                      : 'Click Protect the photo under the image to see if blurring parts of it fools the model.'}
                </p>

                {!selected?.uploaded && !result.privacy?.available && (
                  <p className="muted">
                    There is no protected version of this
                    photo.
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

              Next photo

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
