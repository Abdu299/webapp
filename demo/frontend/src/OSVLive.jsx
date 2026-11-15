
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
  Sparkles,
  Trophy,
  MapPin,
  CircleHelp,
  Globe2
} from 'lucide-react'

import { ISO2_TO_ISO3 } from './countryIso3'
import './OSVLive.css'



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

// --------------------------------------------------
// MAP HELPERS
// --------------------------------------------------


// Highlight only the ground-truth country and matching predictions.
function isCorrectCountry(candidate, truth) {
  if (!candidate || !truth) return false
  return String(candidate).trim().toUpperCase() === String(truth).trim().toUpperCase()
}

const correctCountryStyle = { color: '#16a34a' }

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


  const locked = Boolean(result)
  let winner = 'No ground-truth GPS available'
  if (trueDistance !== null && modelDistance !== null) {
    if (trueDistance < modelDistance) winner = 'You beat the AI!'
    else if (trueDistance > modelDistance) winner = 'AI wins this round'
    else winner = "It's a tie!"
  }

  const topImportance = importance?.clue_importance || []
  const topHeatRatio = topImportance[0]?.heat_ratio || 1
  const continentItems = importance?.continent_clues?.clues || []
  const topContinentEffect = continentItems.find(item => item.effect > 0)?.effect || 1

  return (
    <div className="app osv-page osv-match-main">
      <header className="top">
        <div className="brand">
          <span className="brand-icon"><Globe2 size={23} /></span>
          <div>
            <strong>GeoGuess<span>AI</span></strong>
            <small>VISUAL GEOLOCATION · OSV LIVE</small>
          </div>
        </div>
        <div className="top-right">
          <span className="pill"><span className="green-dot" />LIVE INFERENCE</span>
          <span className="top-sub">IN5310 · GROUP 2</span>
        </div>
      </header>

      {!selected ? (
        <main className="start">
          <div className="intro">
            <span className="eyebrow">OSV LIVE GALLERY</span>
            <h1>Every photo tells<br /><em>a location.</em></h1>
            <p>Choose an OSV-5M photo, place your guess and run GeoViT live. Heatmaps, visual clues and privacy protection are calculated when requested.</p>
          </div>
          <div className="section-title">
            <div><h2>Choose your challenge</h2><p>{photos.length} images in osv_image_galary</p></div>
            <a className="upload" style={{textDecoration:'none'}} href="/"><ArrowLeft size={17}/> Main gallery</a>
          </div>
          {error && <div className="notice">{error}</div>}
          {loading ? <p>Loading images…</p> : photos.length === 0 ? (
            <div className="empty"><h3>No OSV images found</h3><p>Add image files to webapp/osv_image_galary and refresh.</p></div>
          ) : (
            <div className="gallery">
              {photos.map((photo,index) => (
                <button className="photo-card" key={photo.id} onClick={() => open(photo)}>
                  <img src={photo.thumbnail} alt="Mystery location" loading="lazy"/>
                  <span className="photo-shade"/>
                  <span className="photo-tag">OSV IMAGE {String(index+1).padStart(2,'0')}</span>
                  <span className="photo-caption"><strong>Guess the location</strong><ArrowRight size={19}/></span>
                </button>
              ))}
            </div>
          )}
          <div className="footer-note"><ShieldCheck size={16}/> All OSV model operations are performed on demand.</div>
        </main>
      ) : (
        <main className="game">
          <div className="game-head">
            <button className="back" onClick={() => open(null)}><ArrowLeft size={17}/> Back to OSV gallery</button>
            <span className="step">{locked ? 'ROUND REVEALED' : 'ROUND IN PROGRESS'} · OSV LIVE</span>
          </div>

          <div className="game-grid">
            <section className="panel scene-panel">
              <div className="panel-heading">
                <div><span className="eyebrow">THE MYSTERY</span><h2>{blur ? 'Protected photo' : 'Study the photo'}</h2></div>
                <span className="tiny-pill">{blur ? 'PRIVACY PROTECTED' : locked ? 'REVEALED' : 'LOCATION HIDDEN'}</span>
              </div>
              <div className="photo-stage">
                <img className="scene" src={blur?.image_data || original} alt={blur ? 'Protected OSV photo' : 'OSV mystery photo'}/>
                {!blur && showHeat && heatmap && <img className="overlay" src={heatmap.heatmap_data} alt="AI heatmap"/>}
                {!blur && showClues && clues && <img className="overlay" src={clues.overlay_data} alt="Visual clues"/>}
              </div>
              {!locked && <p className="hint"><CircleHelp size={16}/> Look for road signs, architecture, landscape and street markings.</p>}

              {locked && !blur && (
                <>
                  <div className="toggles">
                    <button className={showClues ? 'toggle active':'toggle'} disabled={!!busy} onClick={getClues}>
                      <Layers size={16}/> Visual clues <span>{showClues ? 'ON' : 'OFF'}</span>
                    </button>
                    <button className={showHeat ? 'toggle active':'toggle'} disabled={!!busy} onClick={getHeatmap}>
                      <Eye size={16}/> AI heatmap <span>{showHeat ? 'ON' : 'OFF'}</span>
                    </button>
                  </div>
                  <div className="subheading">Visual clues</div>
                  {clues?.clues?.length ? (
                    <div className="clue-list">
                      {[...clues.clues].filter(c => c.share_pct >= 1).sort((a,b) => b.share_pct-a.share_pct).map(c => (
                        <div className="clue" key={c.name}><i style={{background:c.color}}/><span>{c.name}</span><strong>{c.share_pct}%</strong></div>
                      ))}
                    </div>
                  ) : <p className="muted">Click Visual clues to calculate segmentation for this photo.</p>}

                  <div className="subheading">What the model relied on</div>
                  {!importance && <button className="secondary" disabled={!!busy} onClick={getImportance}><Layers size={16}/> Calculate clue importance</button>}
                  {importance && (topImportance.length ? (
                    <>
                      <p className="muted">How much the model relied on each visual region (1× = an average spot).</p>
                      {topImportance.slice(0,5).map(c => (
                        <div className="rank" key={c.name}>
                          <strong>{c.name}</strong>
                          <div className="bar"><i style={{width:`${Math.min(100, Math.max(4, 100*c.heat_ratio/topHeatRatio))}%`}}/></div>
                          <b>{Number(c.heat_ratio).toFixed(1)}×</b>
                        </div>
                      ))}
                    </>
                  ) : <p className="muted">No significant clue importance was detected.</p>)}
                  {importance?.continent_clues && <>
                    <div className="subheading">What gives {importance.continent} away</div>
                    <p className="muted">Compared with randomly hidden regions across {importance.continent_clues.photos} street-view photos.</p>
                    {continentItems.filter(c => c.effect > 0).slice(0,3).map(c => (
                      <div className="rank" key={c.name} style={{opacity:c.significant ? 1 : .45}}>
                        <strong>{c.name}</strong>
                        <div className="bar"><i style={{width:`${Math.min(100,Math.max(0,100*c.effect/topContinentEffect))}%`}}/></div>
                        <b>−{Math.round(c.effect*100)} pts</b>
                      </div>
                    ))}
                  </>}
                </>
              )}

              {locked && (
                <div className="privacy-flow">
                  {!blur ? (
                    <button className="primary privacy-action" onClick={protect} disabled={!!busy}>
                      <ShieldCheck size={18}/> Protect image <ArrowRight size={18}/>
                    </button>
                  ) : !protectedResult ? (
                    <>
                      <div className="privacy-info"><ShieldCheck size={17}/> The privacy model generated a protected image. Test it to see how GeoViT predicts now.</div>
                      <button className="primary privacy-action" onClick={testProtected} disabled={!!busy}>
                        <Sparkles size={18}/> Test protected image <ArrowRight size={18}/>
                      </button>
                      <button className="privacy-link" onClick={() => {setBlur(null);setProtectedResult(null)}}>Show original image</button>
                    </>
                  ) : (
                    <div className="privacy-reveal">
                      <span className="eyebrow">PROTECTED IMAGE · AI TEST RESULTS</span>
                      <div className="privacy-answer">
                        <div><small>TRUE LOCATION</small><strong style={result?.truth?.country ? correctCountryStyle : undefined}>{result?.truth?.country || 'Unknown'}</strong></div>
                        <div><small>AI PREDICTED AFTER PROTECTION</small><strong style={isCorrectCountry(protectedResult.model?.country, result?.truth?.country) ? correctCountryStyle : undefined}>{protectedResult.model?.country || 'Unknown'}</strong></div>
                      </div>
                      <div className="privacy-stats">
                        <div><small>PROTECTED AI ERROR</small><strong>{fmt(protectedDistance)}</strong></div>
                        <div><small>AI GUESS MOVED</small><strong>{fmt(moved)}</strong></div>
                        <div><small>IMAGE QUALITY · SSIM</small><strong>{blur.ssim ?? 'Unavailable'}</strong></div>
                      </div>
                      <p className="disclaimer">Blurred area: {blur.blurred_area_pct ?? '—'}%. GeoViT predicts countries, not precise coordinates.</p>
                      <button className="privacy-link" onClick={() => {setBlur(null);setProtectedResult(null)}}>← Show original image</button>
                    </div>
                  )}
                </div>
              )}
            </section>

            <section className="panel map-panel">
              <div className="panel-heading">
                <div><span className="eyebrow">THE CHALLENGE</span><h2>{locked ? 'The big reveal' : 'Where was it taken?'}</h2></div>
                <MapPin color="#64748b" size={23}/>
              </div>
              <p className="map-intro">{locked ? 'Compare your location with the live AI prediction.' : 'Click anywhere on the map to place your guess. Drag your pin to refine it.'}</p>
              <GuessMap guess={guess} onGuess={setGuess} locked={locked} result={result} tested={!!protectedResult} privacyResult={protectedResult}/>
              {!locked ? (
                <button className="primary" onClick={reveal} disabled={!guess || !!busy}>
                  Lock in my guess · Run GeoViT <ArrowRight size={18}/>
                </button>
              ) : (
                <>
                  <div className="verdict">
                    <Trophy size={21}/>
                    <div>
                      <strong>{winner}</strong>
                      {trueDistance !== null && modelDistance !== null && (
                        <small>{fmt(Math.abs(trueDistance-modelDistance))} {trueDistance < modelDistance ? 'closer than the AI' : trueDistance > modelDistance ? 'closer for the AI' : 'difference'}</small>
                      )}
                    </div>
                  </div>
                  <div className="metrics">
                    <div><small>YOUR DISTANCE</small><strong>{fmt(trueDistance)}</strong></div>
                    <div><small>AI DISTANCE</small><strong>{fmt(modelDistance)}</strong></div>
                  </div>
                  <div className="answer">
                    <div><small>ACTUAL LOCATION</small><strong style={result.truth?.country ? correctCountryStyle : undefined}>{result.truth?.country || 'Unspecified'}</strong></div>
                    <div><small>AI PREDICTION</small><strong style={isCorrectCountry(result.model?.country, result.truth?.country) ? correctCountryStyle : undefined}>{result.model?.country || 'Unspecified'}</strong></div>
                  </div>
                  <p className="disclaimer">AI pins represent countries rather than exact predicted GPS locations.</p>
                </>
              )}
            </section>
          </div>

          {locked && (
            <div className="details-grid">
              <section className="panel result-panel">
                <div className="panel-heading"><div><span className="eyebrow">MODEL CONFIDENCE</span><h2>Top 3 countries</h2></div><Sparkles size={20}/></div>
                {result.top3?.length ? result.top3.slice(0,3).map((country,index) => (
                  <div className="rank" key={country.country || index} style={isCorrectCountry(country.country, result.truth?.country) ? {background:'#f0fdf4',borderRadius:9,paddingLeft:10,paddingRight:10} : undefined}>
                    <span>0{index+1}</span>
                    <strong style={isCorrectCountry(country.country, result.truth?.country) ? correctCountryStyle : undefined}>{country.country}</strong>
                    <div className="bar"><i style={{width:`${Math.max(0,Math.min(100,country.probability_pct ?? 0))}%`, background:isCorrectCountry(country.country, result.truth?.country) ? '#16a34a' : undefined}}/></div>
                    <b>{Number(country.probability_pct ?? 0).toFixed(1)}%</b>
                  </div>
                )) : <p className="muted">Country confidence data unavailable.</p>}
              </section>
              <section className="panel result-panel">
                <div className="panel-heading"><div><span className="eyebrow">PRIVACY EXPERIMENT</span><h2>Can we hide the clues?</h2></div><ShieldCheck size={23}/></div>
                <p className="map-intro">
                  {protectedResult ? 'The new AI prediction is shown under the photo. The purple pin shows the protected prediction.' : blur ? 'The image has been protected live. Click Test protected image to see the new AI result.' : 'Click Protect image below the photo to run the privacy model live.'}
                </p>
                <p className="muted">OSV processing is on demand, not precomputed.</p>
              </section>
            </div>
          )}
          {busy && <div className="osv-busy"><span className="osv-spinner"/>{busy}</div>}
          {error && <div className="osv-error" role="alert">{error}</div>}
          {locked && <button className="next" onClick={() => open(null)}><RotateCcw size={17}/> Next photo / new visitor <ArrowRight size={17}/></button>}
        </main>
      )}
      <footer className="foot">GEOLOCATION · INTERPRETABILITY · PRIVACY <span>UNIVERSITY OF OSLO · IN5310</span></footer>
    </div>
  )
}
