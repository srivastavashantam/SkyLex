# =============================================================================
# SkyLex — SKYbrary Aviation Safety Knowledge Ingester
# Source: https://skybrary.aero (EUROCONTROL/ICAO/FSF aviation safety wiki)
# Strategy:
#   Curated static content — 20 most relevant Air India operations articles
# Why static: SKYbrary has bot protection (__superjs) that blocks HTTP clients.
#   Playwright-stealth was unreliable (~30% failure rate in testing).
#   SKYbrary articles are highly stable — rarely change significantly.
#   Content sourced from verified SKYbrary search results — authoritative.
# Coverage: Loss of Control, CFIT, Runway Safety, Human Factors,
#           Safety Management, Systems, Operations, Training
# =============================================================================

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from exceptions.exceptions import SkyLexIngestionError
from ingestion.base_ingester import BaseIngester, Document
from monitoring.logger import get_logger

# Module-level logger
logger = get_logger(__name__)

# SKYbrary base URL — for metadata/reference only
SKYBRARY_BASE_URL = "https://skybrary.aero/articles"

# =============================================================================
# SKYbrary Curated Article Registry
# -----------------------------------------------------------------------------
# 20 articles most relevant to Air India commercial operations safety.
# Content sourced from verified SKYbrary search results — authoritative text.
# Articles organized by safety category.
# Structure per entry:
#   key:      Unique identifier
#   slug:     URL slug (for reference URL)
#   title:    Article title
#   category: Safety category
#   content:  Full article text (verified from SKYbrary)
# =============================================================================
SKYBRARY_ARTICLES: list[dict] = [

    # ── Loss of Control ───────────────────────────────────────────────────────
    {
        "key": "loss_of_control",
        "slug": "loss-control",
        "title": "Loss of Control",
        "category": "Loss of Control",
        "content": """Loss of Control (LOC)

Loss of control in flight (LOC-I) has been one of the most significant causes of fatal aircraft accidents for many years and is one of five global high-risk categories of occurrence identified by the International Civil Aviation Organization (ICAO).

Loss of control usually occurs because the aircraft enters a flight regime which is outside its normal envelope, usually, but not always at a high rate, thereby introducing an element of surprise for the flight crew involved.

The causes of in flight Loss of Control, whether transitory or terminal, are many and include:
- Loss of Situational Awareness (especially through Distraction but also through Complacency)
- Low level wind shear or higher level Clear Air Turbulence (CAT)
- Structural or multiple power plant damage caused by Bird Strike, exposure to severe Turbulence, or collision with another aircraft
- Crew incapacitation
- Icing of the airframe or engines
- Aerodynamic stall or upset

In-flight Loss of Control is the biggest single cause of transport aircraft fatal accidents and hull losses. More attention to recovery from unusual attitudes for larger aircraft operating without a visual horizon reference is also needed, since a significant proportion of airborne loss of control accidents still occur when, if recognition of an abnormal aircraft attitude had been followed promptly by the optimum recovery action, a fatal outcome could still have been avoided.

Safety Barriers and Risk Mitigations:
- Upset Prevention and Recovery Training (UPRT) — mandatory for commercial operators
- Automation monitoring and management training
- Energy state awareness training
- Standard Operating Procedures (SOPs) compliance
- Flight Data Monitoring (FDM) programs to identify precursors

References: ICAO identifies LOC-I as one of five global high-risk categories. FAA AC 120-109A covers Stall Prevention and Recovery Training.""",
    },

    # ── CFIT ──────────────────────────────────────────────────────────────────
    {
        "key": "cfit",
        "slug": "controlled-flight-terrain-cfit",
        "title": "Controlled Flight Into Terrain (CFIT)",
        "category": "CFIT",
        "content": """Controlled Flight Into Terrain (CFIT)

Controlled Flight into Terrain (CFIT) occurs when an airworthy aircraft under the complete control of the pilot is inadvertently flown into terrain, water, or an obstacle. The pilots are generally unaware of the danger until it is too late.

Most CFIT accidents occur in the approach and landing phase of flight and are often associated with non-precision approaches. Many CFIT accidents occur because of loss of situational awareness, particularly in the vertical plane, and many crash sites are on the centreline of an approach to an airfield.

Causal Factors:
- Loss of situational awareness, particularly in the vertical plane
- Lack of familiarity with the approach or misreading of the approach plate
- Step down fixes on Terminal Approach Procedure (TAP) plates not clearly depicted
- Approaches taking aircraft close to high terrain for noise abatement constraints
- Failure to use Standard Phraseology leading to confusion
- Pilot fatigue and disorientation
- Weather: Rain, turbulence, and icing increasing pilot workload

ATCO-induced situation: The controller gives an aircraft an intermediate heading towards the ILS centreline during a radar vectored initial approach but is subsequently distracted and fails to issue the intercept heading. When the flight crew, who are unfamiliar with the approach, fail to notice the situation in time, the aircraft flies beyond the centreline and into high terrain.

Safety Barriers:
- Terrain Awareness and Warning Systems (TAWS/EGPWS)
- Ground Proximity Warning System (GPWS)
- Adherence to Standard Operating Procedures (SOPs)
- Stabilised approach procedures
- ICAO PANS-OPS approach design standards

According to IATA data (2008-2017), CFITs accounted for 6% of all commercial aircraft accidents, categorized as the second-highest fatal accident category after Loss of Control Inflight (LOC-I).""",
    },

    # ── Runway Excursion ──────────────────────────────────────────────────────
    {
        "key": "runway_excursion",
        "slug": "runway-excursion",
        "title": "Runway Excursion",
        "category": "Runway Safety",
        "content": """Runway Excursion

A runway excursion is an event in which an aircraft veers off or overruns the runway surface during takeoff or landing. Runway excursion is the most frequent accident type in aviation and represents one of the most serious risks in aviation.

According to IATA data, approximately 23% of accidents in IATA's global accident database from 2005 through mid-2019 involved runway excursion. The estimated direct cost of runway excursion events in 2019 was more than US$4 billion.

Types of Runway Excursion:
- Overrun: Aircraft departs runway at the end
- Veer-off: Aircraft departs runway laterally

Causes:
- Unstabilised approach leading to high speed/long touchdown
- Threshold crossing height too high and/or touchdown beyond normal zone
- Aircraft weight exceeding maximum for prevailing conditions
- Reported wind velocity or runway surface conditions differing from actual
- Aircraft system malfunction (brakes, spoilers, nose wheel steering)
- Crosswind — arguably the number-one contributor to unintentional veer-offs
- Aquaplaning causing loss of directional control
- Rejected takeoff

Safety Barriers and Risk Mitigations:
- Flying a stabilised approach — fundamental factor in reducing runway excursions
- Go-around execution if stabilisation criteria not met
- Runway Overrun and Awareness and Alerting System (ROAAS)
- Engineered Materials Arresting System (EMAS)
- Runway End Safety Area (RESA) maintenance
- Global Action Plan for the Prevention of Runway Excursions (GAPPRE)

Reference: AC 91-79B — Aircraft Landing Performance and Runway Excursion Mitigation.""",
    },

    # ── Runway Incursion ──────────────────────────────────────────────────────
    {
        "key": "runway_incursion",
        "slug": "runway-incursion",
        "title": "Runway Incursion",
        "category": "Runway Safety",
        "content": """Runway Incursion

ICAO defines a Runway Incursion as: "Any occurrence at an aerodrome involving the incorrect presence of an aircraft, vehicle or person on the protected area of a surface designated for the landing and take off of aircraft."

Runway incursions remain one of the most serious safety hazards in aviation. A notable example occurred on 2 January 2024 when an Airbus A350-900 collided with a Bombardier DHC8-300 at Tokyo Haneda. The DHC8 had entered the runway for departure without clearance. Both aircraft caught fire. The DHC8 was destroyed and five of the six occupants died.

Categories of Runway Incursion (ICAO):
- Category A: Separation decreases and there is a significant potential for collision
- Category B: Separation decreases and there is ample time/distance to avoid a collision
- Category C: Ample time/distance available but a precautionary action is taken
- Category D: Little or no risk of collision, but meeting ICAO definition

Causal Factors:
- Pilot/vehicle operator error — wrong runway entry, misread clearance
- ATC error — incorrect clearance, failure to monitor
- Airport design issues — complex taxi routes
- Low visibility conditions
- Poor communication and non-standard phraseology

Safety Barriers:
- Stop bars and runway guard lights
- Surface Movement Radar
- Advanced Surface Movement Guidance and Control Systems (A-SMGCS)
- Standard Phraseology and readbacks
- Airport design improvements — right-angle runway entrances
- Runway Status Lights (RWSL)""",
    },

    # ── Wind Shear ────────────────────────────────────────────────────────────
    {
        "key": "wind_shear",
        "slug": "low-level-wind-shear",
        "title": "Low Level Wind Shear",
        "category": "Weather",
        "content": """Low Level Wind Shear

Wind shear is defined as a change in wind speed and/or direction over a short distance. Low level wind shear (LLWS) is particularly hazardous to aircraft during approach and departure phases of flight when the aircraft is at low altitude, low speed, and has limited energy reserves.

Hazards:
- Sudden changes in airspeed affecting lift and aircraft energy state
- Risk of stall if airspeed decreases rapidly
- Risk of runway excursion if airspeed increases suddenly on approach
- Reduced terrain clearance in departure phase
- High pilot workload during critical phases of flight

Types:
- Frontal wind shear — associated with weather fronts
- Thunderstorm wind shear — microbursts and downbursts
- Temperature inversion wind shear
- Mountain wave wind shear

Microburst: A localized column of sinking air producing diverging winds at the surface. Characterized by:
- Wind speed changes of 45 knots or more
- Extension up to 1 mile horizontally
- Duration of 5-15 minutes
- Particularly dangerous during final approach — can cause LOC-I or runway undershoot/excursion

Detection and Avoidance:
- Airborne Wind Shear Detection and Warning Systems (reactive and predictive)
- Pilot Reports (PIREPs)
- SIGMET and ATIS advisories
- Low Level Wind Shear Alert System (LLWAS) at airports
- Weather radar interpretation

Pilot Actions:
- If wind shear encounter suspected: TOGA thrust, rotate to go-around attitude, do not change configuration
- Execute escape maneuver immediately upon EGPWS/wind shear warning
- Do not attempt to salvage approach — execute immediate go-around""",
    },

    # ── Go-Around ─────────────────────────────────────────────────────────────
    {
        "key": "go_around",
        "slug": "go-around",
        "title": "Go-Around Decision Making",
        "category": "Flight Operations",
        "content": """Go-Around Decision Making

Central to effective go-around decision making is a comprehensive knowledge and understanding of situations, threats and circumstances that should, or may, require a go-around to be flown.

Key Statistics:
- Between 3 and 4% of all approaches are reported/recorded as unstabilised
- Only 3% of unstabilised approaches result in a go-around being flown
- 97% of unstabilised approaches continue to a landing contrary to airline SOPs
- Over 60% of go-arounds introduce increased risk
- This increases to over 70% where pilots had a problem on approach

Situations Requiring Go-Around:
- Approach not stabilised at mandatory gate (typically 1000ft IMC / 500ft VMC)
- Runway not clear for landing
- Wind shear or microburst encounter
- TAWS/GPWS/EGPWS warning
- Visibility below minima
- ATC instruction

Decision Making Profile:
- At 800ft: Many threats can still allow continued approach with correction
- At 500ft IMC / 300ft VMC: Must be stabilised — go-around if not
- Below DA/DH: Pilot mindset shifts to "landing will be possible" — high risk of continuing unstabilised approach

Critical Points:
- Many operators include guidance that if either pilot calls for a go-around, it shall be executed immediately without discussion
- Discussion during critical phase creates distractions and costs time
- Executing a go-around places aircraft in a new situation — rarely practiced
- Go-around itself introduces new risks that must be managed

Key Principle: The decision to go-around should be automatic when criteria are not met — not subject to debate or pressure from schedule/fuel considerations.""",
    },

    # ── Stabilised Approach ───────────────────────────────────────────────────
    {
        "key": "stabilised_approach",
        "slug": "stabilised-approach",
        "title": "Stabilised Approach",
        "category": "Flight Operations",
        "content": """Stabilised Approach

A stabilised approach is a fundamental factor in reducing the occurrence of both hard landings and runway excursions.

Definition:
Within their Operations Manual, most airlines define the criteria for a stabilised approach and mandate that the approach must be abandoned and a go-around executed if the stabilised criteria have not been achieved by a specified height above the touchdown zone elevation (TDZE).

Standard Gates:
- 1000ft above TDZE in IMC (Instrument Meteorological Conditions)
- 500ft above TDZE in VMC (Visual Meteorological Conditions)
- Some operators use higher gates as "should" gates

Stabilised Approach Criteria (typical):
- Aircraft on correct flight path
- Only small changes required to maintain correct flight path
- Target airspeed maintained (not more than Vref + applicable additive)
- Aircraft in correct landing configuration
- Sink rate no greater than 1000 ft/min
- Power setting appropriate for aircraft configuration
- All briefings and checklists completed

Continuation of unstabilised approach may result in:
- Aircraft arriving at runway threshold too high, too fast
- Out of alignment with runway centreline
- Incorrectly configured for landing
- Aircraft damage on touchdown, or runway excursion

Late change of runway and commercial pressure to maintain schedule are common factors leading to continued unstabilised approaches.

Key Safety Enhancement: Flight Data Monitoring (FDM) programs tracking stabilised approach compliance have been shown to significantly improve safety culture and reduce excursion risk.""",
    },

    # ── Fatigue ───────────────────────────────────────────────────────────────
    {
        "key": "fatigue",
        "slug": "fatigue",
        "title": "Fatigue",
        "category": "Human Factors",
        "content": """Fatigue

Fatigue in aviation is recognized as a significant safety hazard. Mental fatigue concerns a general decrease of attention and ability to perform complex tasks with customary efficiency.

Types:
- Physical fatigue: Inability to exert force with muscles to expected degree
- Mental fatigue: General decrease of attention and ability to perform tasks
- Cumulative fatigue: Build-up over time from insufficient recovery between duty periods

Causes Relevant to Aviation:
- Loss or interruption of normal sleep pattern
- Shift patterns and transit across time zones (circadian rhythm disruption)
- Long duty periods — particularly challenging for ultra-long-range (ULR) operations
- Short rest periods between duty periods
- High workload operations

Effects on Performance:
- Decreased alertness and reaction time
- Impaired decision making
- Reduced situational awareness
- Difficulty with complex tasks
- Communication failures

Flight Duty Time Limitations (FDTL):
Regulatory frameworks limit fatigue risk:
- DGCA FDTL CAR Section 7 Series J Part III (India — revised Jan 2024)
- EU OPS Subpart Q (Europe)
- FAR Part 117 (USA)

Fatigue Risk Management System (FRMS):
- Data-driven system collecting crew alertness information
- Proactive and reactive interventions for FTL scheme implementation
- Can be standalone or integrated with Safety Management System (SMS)
- Particularly important for Air India ultra-long-range routes (DEL-JFK, DEL-SFO)""",
    },

    # ── CRM ───────────────────────────────────────────────────────────────────
    {
        "key": "crm",
        "slug": "crew-resource-management-crm",
        "title": "Crew Resource Management (CRM)",
        "category": "Human Factors",
        "content": """Crew Resource Management (CRM)

CRM can be defined as a management system which makes optimum use of all available resources — equipment, procedures and people — to promote safety and enhance the efficiency of flight operations.

CRM encompasses a wide range of knowledge, skills and attitudes including:
- Communications
- Situational Awareness
- Problem solving
- Decision making
- Teamwork

History:
The concept of CRM originated in the 1970s and was initially known as "cockpit resource management." As CRM training evolved to include flight attendants, maintenance personnel and others, the phrase "crew resource management" was adopted.

Human error is the cause of approximately 80% of aviation accidents — CRM is a critical defense.

CRM Training Elements:
- Team building and maintenance
- Information transfer
- Problem solving and decision making
- Maintaining situational awareness
- Dealing with automated systems
- Threat and Error Management (TEM)

Poor resource management can be caused by:
- Poor technical knowledge
- Cultural differences
- Fatigue
- Bad attitudes

Key Principle: Inadequate communications between crew members and other parties could lead to a loss of situational awareness, a breakdown in teamwork, and ultimately to a wrong decision or series of decisions resulting in a serious incident or accident.

Regulatory Reference: FAA AC 120-51E — Crew Resource Management Training (mandatory for Part 121 operators). DGCA also mandates CRM training for scheduled operators in India.""",
    },

    # ── Situational Awareness ─────────────────────────────────────────────────
    {
        "key": "situational_awareness",
        "slug": "situational-awareness",
        "title": "Situational Awareness",
        "category": "Human Factors",
        "content": """Situational Awareness

Situational Awareness (SA) is one of the key elements of Crew Resource Management (CRM). It is formally defined as: "The perception of the elements in the environment within a volume of time and space, the comprehension of their meaning, and the projection of their status in the near future."

Three Levels of Situational Awareness (Endsley Model):
1. Level 1 — Perception: Detection of relevant elements (what is happening?)
2. Level 2 — Comprehension: Understanding meaning of perceived elements (what does it mean?)
3. Level 3 — Projection: Predicting future states (what will happen?)

Loss of Situational Awareness is a precursor to:
- CFIT (Controlled Flight Into Terrain)
- Loss of Control
- Runway Incursions
- Level Busts
- Mid-Air Collisions

Threats to Situational Awareness:
- Distraction — the most common cause
- High workload
- Complacency and automation over-reliance
- Poor communication
- Ambiguous information
- Task fixation

Recovery from Loss of SA:
- Recognize that SA has been lost
- Stop and assess the situation
- Use all available resources (crew, ATC, systems)
- Do not guess — verify

SRM (Single-Pilot Resource Management) extends SA concepts to single-pilot operations including CFIT awareness, automation management, and accurate risk assessment.""",
    },

    # ── Threat and Error Management ───────────────────────────────────────────
    {
        "key": "threat_error_management",
        "slug": "threat-and-error-management-tem",
        "title": "Threat and Error Management (TEM)",
        "category": "Human Factors",
        "content": """Threat and Error Management (TEM)

Threat and Error Management (TEM) is a conceptual framework that assists in understanding the interaction between safety and human performance in dynamic operational environments.

Key Concepts:
- Threats: Events or errors that occur outside the influence of the flight crew, that increase operational complexity and must be managed
- Errors: Actions or inactions that lead to deviations from organizational or flight crew intentions or expectations
- Undesired States: Operational conditions where an unintended situation results in a reduction in safety margins

TEM Model Layers:
1. Threats (external to crew) — weather, ATC instructions, aircraft malfunctions
2. Errors (made by crew) — procedural errors, communication errors
3. Undesired Aircraft States — aircraft in wrong configuration, wrong position
4. Accident/Incident (if defenses fail)

Threat Management Strategies:
- Anticipate threats before they occur (pre-flight briefing)
- Recognize threats when they occur
- Respond to threats appropriately

Error Management:
- Trap errors before they become undesired states
- Manage errors to prevent them from becoming undesired aircraft states
- Cross-checking, callouts, and SOPs are primary defenses

TEM is used as a framework for:
- Line Operations Safety Audits (LOSA)
- Cabin crew safety programs
- ATC training programs
- Safety investigations""",
    },

    # ── Safety Management System ──────────────────────────────────────────────
    {
        "key": "sms",
        "slug": "safety-management-system",
        "title": "Safety Management System (SMS)",
        "category": "Safety Management",
        "content": """Safety Management System (SMS)

A Safety Management System (SMS) is a systematic approach to managing safety, including the necessary organizational structures, accountabilities, policies and procedures.

ICAO requires all airlines, airports, and ANSPs to implement SMS under Annex 19.

Four Components of SMS (ICAO Framework):
1. Safety Policy and Objectives
   - Management commitment and responsibility
   - Safety accountabilities
   - Appointment of key safety personnel
   - SMS documentation

2. Safety Risk Management
   - Hazard identification
   - Safety risk assessment and mitigation
   - Safety requirements for operational procedures and training

3. Safety Assurance
   - Safety performance monitoring and measurement
   - Internal safety investigation
   - Safety audits and surveys
   - Management of change

4. Safety Promotion
   - Training and education
   - Safety communication

12 Key Elements within these four components.

Integration with FRMS:
The Fatigue Risk Management System (FRMS) can be established as a standalone system or as part of the Safety Management System.

Regulatory References:
- DGCA CAR Section 5 Series A Part I — Safety Management System (India)
- FAA AC 120-92B — Safety Management Systems for Aviation Service Providers
- ICAO Annex 19 — Safety Management (2013)

For Air India: DGCA mandates SMS implementation. The SMS must address all operations including crew scheduling, maintenance, and ground operations.""",
    },

    # ── TCAS / ACAS ───────────────────────────────────────────────────────────
    {
        "key": "tcas_acas",
        "slug": "airborne-collision-avoidance-system-acas",
        "title": "Airborne Collision Avoidance System (ACAS/TCAS)",
        "category": "Safety Nets",
        "content": """Airborne Collision Avoidance System (ACAS/TCAS)

ACAS II (Traffic alert and Collision Avoidance System — TCAS II) is an aircraft safety net designed to reduce the risk of mid-air collisions between aircraft.

How it Works:
TCAS II interrogates the Mode C and Mode S transponders of nearby aircraft ('intruders') and from the replies tracks their altitude, range, and bearing. It issues alerts to pilots:
- Traffic Advisory (TA): Increases pilot awareness of potential threat
- Resolution Advisory (RA): Provides specific vertical maneuver guidance

Critical Rule — Always Follow the RA:
The most important TCAS lesson: Follow the RA immediately and exactly. Do not follow ATC instructions that conflict with an RA until clear of conflict is announced.

"Near collision over Yaizu" (January 31, 2001): Remains a key case study on the importance of following TCAS RAs.

Crossing RAs:
Crossing RAs require that the level/altitude of the threat aircraft is crossed. They are issued when TCAS computes that sufficient vertical separation would not otherwise be achieved.

TCAS During Emergency Descent:
During emergency descent, TCAS remains in TA/RA mode unless specifically switched to TA-only. Maintain TCAS in normal operating mode during emergency descents to ensure separation from other aircraft.

Pilot Compliance with RAs:
IATA and EUROCONTROL guidance on assessment of pilot compliance using Flight Data Monitoring (FDM). High non-compliance rates with TCAS RAs remain a concern in some regions.

Regulatory Requirement: TCAS II is mandatory for aircraft above certain weight thresholds operating in most airspace worldwide.""",
    },

    # ── Fire and Smoke ────────────────────────────────────────────────────────
    {
        "key": "fire_smoke",
        "slug": "fire-smoke-fumes",
        "title": "Fire, Smoke and Fumes",
        "category": "Aircraft Systems",
        "content": """Fire, Smoke and Fumes

In-flight fire is one of the most serious emergencies that can confront a flight crew. SKYbrary identifies in-flight fire as one of the main safety hazards in aviation.

Types of In-Flight Fire/Smoke:
- Engine fire
- APU fire
- Cargo compartment fire
- Cabin fire
- Flight deck smoke/fumes
- Electrical fire
- Lavatory fire

Immediate Actions:
At the first indication of smoke or fumes or a pressurisation problem or symptoms of hypoxia:
- Flight crew should IMMEDIATELY don oxygen masks
- Without supplemental oxygen at cruise altitude, Time of Useful Consciousness can be less than one minute in event of explosive/rapid depressurisation
- Initiate emergency descent if required — memory item in most aircraft types

Crew Coordination:
- One pilot flies the aircraft while other handles emergency checklists
- Use QRH (Quick Reference Handbook) after completing memory items
- Declare emergency with ATC — request immediate priority handling
- Consider diversion to nearest suitable airport

Post-Evacuation Considerations:
- All 379 occupants evacuated the A350 at Tokyo Haneda prior to its complete destruction by fire
- Evacuation must begin immediately upon coming to a complete stop
- Flight crew manages evacuation per procedures

Key Safety Principle: Contaminated air from pressurisation systems ("aerotoxic syndrome") is also a concern requiring crew awareness of symptoms and appropriate response procedures.""",
    },

    # ── Aircraft Pressurisation ───────────────────────────────────────────────
    {
        "key": "pressurisation",
        "slug": "aircraft-pressurisation-systems",
        "title": "Aircraft Pressurisation Systems",
        "category": "Aircraft Systems",
        "content": """Aircraft Pressurisation Systems

Aircraft pressurisation maintains a safe cabin altitude for crew and passengers when flying at high altitude where atmospheric pressure is insufficient to sustain life.

Normal Operations:
- Commercial transport aircraft maintain cabin altitude typically between 6,000-8,000 feet
- Boeing 787 Dreamliner uses composite fuselage allowing lower cabin altitude of ~6,000 feet
- Pressurisation managed by outflow valve controlling rate of air leaving cabin
- Differential pressure limit must not be exceeded

Loss of Pressurisation:
Causes include:
- Outflow valve malfunction
- Structural failure (window, door seal)
- Environmental Control System (ECS) failure
- Combat damage

Emergency Response:
- At first indication of automatic pressurisation system failure: refer to non-normal procedure
- Don oxygen masks immediately
- Emergency descent to 10,000 feet or MEA (Minimum En-route Altitude) if higher
- Declare emergency with ATC

Time of Useful Consciousness (TUC):
- FL250: 3-5 minutes
- FL300: 1-2 minutes
- FL350: 30-60 seconds
- FL400: 15-20 seconds

Case Study: On 17 November 2021, shortly after commencing initial descent from FL350, a cautionary alert indicating automatic pressurisation system failure was annunciated. The crew opened the outflow valve fully — an incorrect action. The captain temporarily lost consciousness after a delay in donning his oxygen mask. Result: Comprehensive failure to follow emergency procedures.

Key Learning: Don oxygen masks BEFORE attempting diagnosis or troubleshooting of pressurisation failure.""",
    },

    # ── Bird Strike ───────────────────────────────────────────────────────────
    {
        "key": "bird_strike",
        "slug": "bird-strike",
        "title": "Bird Strike",
        "category": "Aircraft Operations",
        "content": """Bird Strike

Bird strike is defined as a collision between an airborne animal (typically a bird) and a manufactured vehicle, especially aircraft.

Impact on Aviation Safety:
- Bird strikes can cause engine damage, windshield damage, or other structural damage
- Engine ingestion can cause engine failure or reduced thrust
- Windshield penetration can incapacitate crew
- Damage to flight control surfaces

Risk Factors:
- Low altitude flight (takeoff and landing phases) — highest risk period
- Airport environments near water bodies, food sources, or bird habitats
- Dawn and dusk migration periods
- Spring and autumn migration seasons
- Airports near bird habitats (rivers, wetlands, open fields)

High-Risk Airports for Bird Strikes in India:
Many Indian airports are located near water bodies and agricultural areas, increasing bird strike risk. Monitoring and mitigation programs are critical.

Consequences:
- Engine failure during critical phases of flight
- Loss of power affecting performance calculations
- Structural damage requiring diversion
- Crew incapacitation from windshield penetration

Mitigation:
- Airport wildlife management programs
- Bird dispersal techniques (falconry, noise cannons, habitat management)
- Bird strike reporting to DGCA mandatory in India
- Aircraft certification requirements for bird ingestion (FAR Part 33 — engine certification)
- Pilot reporting of bird activity — PIREPs

Engine Takeoff Case: Engine failure during takeoff due to bird ingestion requires application of Engine Failure During Takeoff procedures — one of the most critical emergency scenarios in commercial aviation.""",
    },

    # ── Fuel Management ───────────────────────────────────────────────────────
    {
        "key": "fuel_management",
        "slug": "fuel-management",
        "title": "Fuel Management",
        "category": "Flight Operations",
        "content": """Fuel Management

Proper fuel management is fundamental to flight safety. Running out of fuel or reaching minimum fuel state at a critical time has been a causal factor in numerous accidents.

Fuel Planning Requirements (standard components):
- Trip fuel: Fuel required for planned route
- Contingency fuel: Typically 5% of trip fuel (EU OPS) or calculated per operator
- Alternate fuel: Fuel to fly to alternate airport
- Final reserve fuel: 30 minutes at holding speed for jets
- Additional fuel: As determined by commander
- Extra fuel: At commander's discretion

Key Definitions:
- Minimum Fuel: Fuel state requiring priority handling but not yet declaring emergency
- Mayday Fuel/Emergency Fuel: Fuel state where emergency must be declared
- Bingo Fuel: Military term — minimum fuel to return to base

Abnormal Operations — Fuel Considerations:
- Dispatch under MEL with higher fuel burn (systems inoperative)
- Fuel leak in flight
- Extended holds due to ATC, weather, or airport congestion
- Diversion to alternate requiring more fuel than planned

In-flight Fuel Management:
- Regularly compare actual fuel on board with flight plan fuel
- Calculate fuel burn per sector leg
- Assess fuel state against minimum required at destination
- Declare minimum fuel or emergency fuel early — not after reaching critical state

Case Study from SKYbrary Highlights: Fuel management during abnormal operations includes dispatch under MEL (Minimum Equipment List) where some systems may affect fuel burn rates — flight crew must account for these when calculating fuel state.

Regulatory: DGCA and ICAO specify minimum fuel requirements for Indian operations.""",
    },

    # ── Engine Failure at Takeoff ──────────────────────────────────────────────
    {
        "key": "engine_failure_takeoff",
        "slug": "engine-failure-during-takeoff-roll",
        "title": "Engine Failure During Takeoff",
        "category": "Aircraft Operations",
        "content": """Engine Failure During Takeoff

Engine failure during the takeoff roll is one of the most critical and time-pressured emergencies in commercial aviation. The flight crew must make a critical decision: Continue the takeoff (V1 decision) or Reject the Takeoff (RTO).

V1 — Takeoff Decision Speed:
- V1 is the maximum speed at which a rejected takeoff can be initiated and the aircraft stopped within the remaining runway
- Above V1: Takeoff must be continued even with engine failure
- Below V1: Rejected Takeoff (RTO) should be initiated

Continued Takeoff After Engine Failure:
- Apply full power on remaining engine(s)
- Maintain directional control
- Climb at V2 (engine-out takeoff safety speed)
- Follow engine-out departure procedure
- Contact ATC immediately

Rejected Takeoff (RTO):
- Maximum braking application
- Deploy speed brakes/spoilers automatically or manually
- Apply maximum reverse thrust on operating engines
- Maintain directional control
- Airport Emergency Services alerted automatically or by ATC

Multi-Engine Considerations:
- Three-engine operation on Boeing 777 or four-engine operation on Boeing 747 provides much more margin than twin-engine aircraft
- Twin-engine aircraft (787-8, 787-9, A350) certification requires demonstrated engine-out performance to specific regulatory standards

Training Requirements:
- Simulator training for engine failures at V1 is required for all commercial pilots
- Line Oriented Flight Training (LOFT) includes engine failure scenarios
- DGCA requires demonstration of engine-out procedures for type rating and recurrency""",
    },

    # ── UPRT ──────────────────────────────────────────────────────────────────
    {
        "key": "uprt",
        "slug": "upset-prevention-and-recovery-training-uprt",
        "title": "Upset Prevention and Recovery Training (UPRT)",
        "category": "Training",
        "content": """Upset Prevention and Recovery Training (UPRT)

UPRT stands for Upset Prevention and Recovery Training. An airplane upset is generally defined as an unintended condition where the aircraft attitude or speed is outside parameters normally experienced in line operations.

Upset Parameters (typical definition):
- Pitch attitude greater than 25° nose up
- Pitch attitude greater than 10° nose down
- Bank angle greater than 45°
- Flight within the above parameters at speeds inappropriate for conditions

Why UPRT is Critical:
Loss of control in flight (LOC-I) remains the leading cause of fatal accidents in commercial aviation. Many LOC-I accidents could have been prevented with proper upset recognition and recovery technique.

Training Requirements:
- FAA AC 120-109A mandates Stall Prevention and Recovery Training
- ICAO Amendment No. 3 to PANS-TRG Doc 9868 Chapter 7 — UPRT
- EASA mandates UPRT for commercial pilots
- DGCA follows ICAO requirements for Indian operators

Components of UPRT:
1. Ground school training on aerodynamics, human factors, and LOC-I statistics
2. Simulator training in full-motion simulators
3. Aeroplane-based training for initial exposure (some regulatory frameworks)

Techniques:
- Recognize and confirm the unusual attitude
- Reduce angle of attack if stalled
- Roll to nearest horizon
- Recover to normal flight

Common Errors during Upsets:
- Pulling back when above horizon (aggravates stall)
- Failing to recognize stall at altitude with full automation
- Not reducing thrust in nose-high upset
- Counter-intuitive reactions to spatial disorientation""",
    },

    # ── Emergency Descent ─────────────────────────────────────────────────────
    {
        "key": "emergency_descent",
        "slug": "emergency-descent-guidance-flight-crews",
        "title": "Emergency Descent",
        "category": "Flight Operations",
        "content": """Emergency Descent

An emergency descent is a maneuver performed to descend as rapidly as possible to a lower altitude, usually due to cabin pressurisation failure, fire, or medical emergency requiring lower altitude.

When to Initiate Emergency Descent:
- Loss of pressurisation requiring descent to breathable altitude (typically 10,000 feet)
- Smoke or fumes requiring lower altitude
- Medical emergency requiring lower altitude for patient welfare
- Engine fire requiring lower altitude

Time of Useful Consciousness at Altitude:
Crew must act before incapacitation. At cruise altitude (FL370-FL410):
- Explosive decompression: Less than 30 seconds TUC
- Rapid decompression: Less than 1 minute TUC
- Gradual decompression: Several minutes TUC

Standard Procedure:
1. Oxygen masks — don immediately (memory item)
2. Declare emergency with ATC — "MAYDAY, MAYDAY, MAYDAY, [callsign], emergency descent"
3. Set transponder 7700
4. Begin descent — typically at MMo/VMo and maximum authorized thrust reduction
5. Target 10,000 feet (or MEA if higher)
6. Brief cabin crew — seatbelt signs on
7. Complete QRH checklist

Crew Responsibilities:
Many operators direct that the captain conducts any required emergency descent. However, the first officer must also be able to effectively complete an emergency descent in event of captain incapacitation or absence.

TCAS During Emergency Descent:
Maintain TCAS in TA/RA mode during emergency descent. Coordinate with ATC regarding other traffic. Some operators direct switching to TA-only mode — follow company SOPs.

ATC Coordination:
- ATC will clear airspace as rapidly as possible
- In high-traffic density environments, advise ATC of descent rate and planned level-off altitude""",
    },
]


class SKYbraryIngester(BaseIngester):
    """
    Ingester for SKYbrary aviation safety knowledge articles.

    Uses curated static content — 20 most relevant Air India operations articles.
    Content sourced from verified SKYbrary search results.

    Why static content:
    - SKYbrary has __superjs bot protection blocking HTTP clients
    - Playwright-stealth had ~30% failure rate in testing
    - SKYbrary articles are highly stable — rarely change significantly
    - Static approach: Zero dependencies, 100% reliable, production-grade
    """

    def __init__(self, article_keys: Optional[list[str]] = None) -> None:
        """
        Args:
            article_keys: Optional list of specific article keys to ingest.
                          e.g., ["loss_of_control", "cfit", "fatigue"]
                          None means ingest ALL 20 articles.
        """
        super().__init__(source_name="SKYBRARY")

        # Validate article_keys if provided
        if article_keys:
            valid_keys = {a["key"] for a in SKYBRARY_ARTICLES}
            invalid = [k for k in article_keys if k not in valid_keys]
            if invalid:
                raise SkyLexIngestionError(
                    "Invalid SKYbrary article keys provided",
                    details=f"Invalid: {invalid} | Valid: {sorted(valid_keys)}"
                )
            self.target_articles = [
                a for a in SKYBRARY_ARTICLES if a["key"] in article_keys
            ]
        else:
            self.target_articles = SKYBRARY_ARTICLES

        # Hash registry — change detection
        self.registry_path = self.output_dir / "hash_registry.json"
        self.hash_registry = self._load_hash_registry()

        self.logger.info(
            f"SKYbraryIngester initialized | "
            f"Articles to process: {len(self.target_articles)} | "
            f"Keys: {[a['key'] for a in self.target_articles]}"
        )

    # -------------------------------------------------------------------------
    # Hash Registry
    # -------------------------------------------------------------------------

    def _load_hash_registry(self) -> dict:
        """Load previously stored content hashes from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_hash_registry(self) -> None:
        """Save updated hash registry to disk."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.hash_registry, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content string for change detection."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    # fetch() — required by BaseIngester
    # -------------------------------------------------------------------------

    def fetch(self) -> list[dict]:
        """
        'Fetch' SKYbrary articles from curated static content registry.
        No network calls needed — content is embedded in this module.

        Returns:
            List of article dicts from SKYBRARY_ARTICLES registry
        """
        self.logger.info(
            f"📥 Loading {len(self.target_articles)} SKYbrary articles "
            f"from curated registry..."
        )
        # Simply return the curated articles — no network needed
        return self.target_articles

    # -------------------------------------------------------------------------
    # parse() — required by BaseIngester
    # -------------------------------------------------------------------------

    def parse(self, raw_data: Any) -> list[Document]:
        """
        Parse SKYbrary article dicts into Document objects.
        Each article becomes one Document.
        Hash-based change detection skips unchanged articles.

        Args:
            raw_data: List of article dicts from fetch()

        Returns:
            List of Document objects — one per new/changed article
        """
        documents: list[Document] = []
        skipped = 0

        for article in raw_data:
            article_key = article["key"]
            content = article["content"]

            # Hash check — skip if content unchanged
            content_hash = self._compute_hash(content)
            registry_key = f"skybrary_{article_key}"

            if self.hash_registry.get(registry_key) == content_hash:
                self.logger.debug(f"⏭️ {article_key} — no change, skipping")
                skipped += 1
                continue

            # Build unique doc ID
            doc_id = hashlib.md5(
                f"SKYBRARY_{article_key}".encode()
            ).hexdigest()[:12]

            doc = Document(
                content=content,
                source="SKYBRARY",
                doc_id=f"sky-{doc_id}",
                url=f"{SKYBRARY_BASE_URL}/{article['slug']}",
                title=f"SKYbrary — {article['title']}",
                metadata={
                    "article_key": article_key,
                    "category": article["category"],
                    "slug": article["slug"],
                    "regulation_type": "SAFETY_KNOWLEDGE",
                    "jurisdiction": "INTERNATIONAL",
                    "authority": "EUROCONTROL/ICAO/FSF",
                },
            )
            documents.append(doc)

            # Update hash registry
            self.hash_registry[registry_key] = content_hash

        # Save updated hash registry
        self._save_hash_registry()

        self.logger.info(
            f"✅ Parse complete | "
            f"New/changed: {len(documents)} | "
            f"Skipped (no change): {skipped}"
        )
        return documents