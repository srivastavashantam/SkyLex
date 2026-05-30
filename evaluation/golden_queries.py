"""
evaluation/golden_queries.py

SkyLex — 75 Golden Queries for RAGAS Evaluation.

Har source se 15 queries × 5 types:
  1. factual    — Specific rule/section/number lookup
  2. procedural — Steps ya process kya hai
  3. cross_ref  — Ek regulation doosre se kaise related hai
  4. edge_case  — Exception, special condition, boundary case
  5. comparative — Do concepts/rules/standards ke beech difference

Ground truths ONLY from actual corpus content — no hallucination.
All answers verified against gq_reference_*.json files.

Format: list of dicts, each dict = one golden query.
Fields:
  query_id         : Unique ID e.g. "CFR-001"
  source           : "FAA_CFR" | "FAA_AD" | "FAA_AC" | "DGCA_CAR" | "SKYBRARY"
  query_type       : "factual" | "procedural" | "cross_ref" | "edge_case" | "comparative"
  question         : The actual query string
  ground_truth     : Correct answer (verified from corpus)
  relevant_doc_ids : List of doc_ids that contain the answer
  difficulty       : "easy" | "medium" | "hard"
  notes            : Optional notes for evaluator
"""

GOLDEN_QUERIES: list[dict] = [

    # =========================================================================
    # SOURCE: FAA_CFR (14 CFR) — 15 queries
    # =========================================================================

    {
        "query_id":        "CFR-001",
        "source":          "FAA_CFR",
        "query_type":      "factual",
        "question":        "Under 14 CFR Part 117, what is the applicability of flight and duty limitations?",
        "ground_truth":    "Part 117 prescribes flight and duty limitations and rest requirements for all flightcrew members and certificate holders conducting passenger operations under part 121 of this chapter.",
        "relevant_doc_ids": ["cfr-046af492fb4a"],
        "difficulty":      "easy",
        "notes":           "Direct applicability lookup from § 117.1",
    },
    {
        "query_id":        "CFR-002",
        "source":          "FAA_CFR",
        "query_type":      "factual",
        "question":        "What does 14 CFR Part 39 define as airworthiness directives?",
        "ground_truth":    "FAA's airworthiness directives are legally enforceable rules that apply to the following products: aircraft, aircraft engines, propellers, and appliances.",
        "relevant_doc_ids": ["cfr-bd2da1da22a2"],
        "difficulty":      "easy",
        "notes":           "Direct definition from § 39.3",
    },
    {
        "query_id":        "CFR-003",
        "source":          "FAA_CFR",
        "query_type":      "factual",
        "question":        "What is the purpose of 14 CFR Part 5 — Safety Management Systems?",
        "ground_truth":    "Part 5 establishes requirements for Safety Management Systems (SMS). It applies to certificate holders operating under part 121, and covers safety policy, safety risk management, safety assurance, and safety promotion components.",
        "relevant_doc_ids": ["cfr-3a9bbbf95f1a"],
        "difficulty":      "easy",
        "notes":           "§ 5.1 applicability",
    },
    {
        "query_id":        "CFR-004",
        "source":          "FAA_CFR",
        "query_type":      "procedural",
        "question":        "What are the general requirements under 14 CFR Part 43 for maintenance record entries?",
        "ground_truth":    "Part 43 governs maintenance, preventive maintenance, rebuilding, and alteration of aircraft. Under § 43.9, each person who maintains an aircraft shall make a maintenance record entry describing the work performed, date, and signature. The record must include a description of work performed, the date of completion, the name of the person performing the work, and a certificate number.",
        "relevant_doc_ids": ["cfr-cc8e13c82903"],
        "difficulty":      "medium",
        "notes":           "§ 43.9 maintenance records",
    },
    {
        "query_id":        "CFR-005",
        "source":          "FAA_CFR",
        "query_type":      "procedural",
        "question":        "What does 14 CFR Part 119 require for certification of air carriers and commercial operators?",
        "ground_truth":    "Part 119 applies to each person operating or intending to operate civil aircraft as an air carrier or commercial operator. It requires obtaining an Air Carrier Certificate or Operating Certificate, designating management personnel including a Director of Operations and Director of Maintenance, and complying with applicable operating rules under parts 121 or 135.",
        "relevant_doc_ids": ["cfr-21c4cca85996"],
        "difficulty":      "medium",
        "notes":           "§ 119.1 applicability and § 119.65 management personnel",
    },
    {
        "query_id":        "CFR-006",
        "source":          "FAA_CFR",
        "query_type":      "procedural",
        "question":        "Under 14 CFR Part 145, what are the requirements for a certificated repair station?",
        "ground_truth":    "Part 145 describes how to obtain a repair station certificate and the rules a certificated repair station must follow for maintenance, preventive maintenance, or alterations. Requirements include facility adequacy (§ 145.103), personnel qualifications, equipment and materials, inspection systems (§ 145.211), training programs (§ 145.163), and a Repair Station Manual.",
        "relevant_doc_ids": ["cfr-ebc2bc4aa1d2"],
        "difficulty":      "medium",
        "notes":           "Subparts A through D of Part 145",
    },
    {
        "query_id":        "CFR-007",
        "source":          "FAA_CFR",
        "query_type":      "cross_ref",
        "question":        "How does 14 CFR Part 121 reference Part 117 for flightcrew rest requirements?",
        "ground_truth":    "Part 121 operating rules for domestic, flag, and supplemental operations reference Part 117 for flight and duty limitations applicable to flightcrew members. § 121.543 specifically references Part 117 requirements. Certificate holders conducting passenger operations under Part 121 must comply with Part 117 limitations including flight duty period, rest period, and flight time limits.",
        "relevant_doc_ids": ["cfr-7d5ded315310", "cfr-046af492fb4a"],
        "difficulty":      "medium",
        "notes":           "Cross-reference between Part 121 and Part 117",
    },
    {
        "query_id":        "CFR-008",
        "source":          "FAA_CFR",
        "query_type":      "cross_ref",
        "question":        "What is the relationship between 14 CFR Part 43 maintenance records and Part 91 aircraft airworthiness requirements?",
        "ground_truth":    "Part 43 (§ 43.9, § 43.11) requires maintenance record entries for all work performed, while Part 91 (§ 91.405) requires aircraft owners and operators to ensure their aircraft is maintained in an airworthy condition. Part 91 (§ 91.417) requires maintenance records to be retained — 24 months for routine maintenance entries and permanently for major repairs and alterations. Part 43 defines the content of records; Part 91 defines the retention obligation.",
        "relevant_doc_ids": ["cfr-cc8e13c82903", "cfr-8deb48bd94de"],
        "difficulty":      "hard",
        "notes":           "Integration of record-keeping requirements",
    },
    {
        "query_id":        "CFR-009",
        "source":          "FAA_CFR",
        "query_type":      "cross_ref",
        "question":        "How does 14 CFR Part 39 relate to Part 43 for compliance with airworthiness directives?",
        "ground_truth":    "Part 39 establishes the legal framework for FAA's airworthiness directives and makes them legally enforceable rules (§ 39.5, § 39.7). Part 43 governs the actual performance of maintenance required by those ADs — who may perform the work (§ 43.3), how it must be performed (§ 43.13), and how it must be recorded (§ 43.9). An AD issued under Part 39 creates the obligation; Part 43 defines the execution and documentation standards.",
        "relevant_doc_ids": ["cfr-bd2da1da22a2", "cfr-cc8e13c82903"],
        "difficulty":      "hard",
        "notes":           "Regulatory framework interconnection",
    },
    {
        "query_id":        "CFR-010",
        "source":          "FAA_CFR",
        "query_type":      "edge_case",
        "question":        "Under 14 CFR Part 91, what are the general operating rules for an aircraft operated outside the United States?",
        "ground_truth":    "Under § 91.703, operations of civil aircraft of U.S. registry outside the United States must comply with Part 91 and the regulations of the foreign country where the aircraft is operated. The pilot in command must comply with ICAO standards where they are more stringent than Part 91. § 91.701 specifies that Subpart H applies to operations outside the U.S., and foreign aircraft operating in the U.S. must comply with § 91.703.",
        "relevant_doc_ids": ["cfr-8deb48bd94de"],
        "difficulty":      "hard",
        "notes":           "International operations edge case",
    },
    {
        "query_id":        "CFR-011",
        "source":          "FAA_CFR",
        "query_type":      "edge_case",
        "question":        "When does 14 CFR Part 117 NOT apply to flightcrew operations under Part 121?",
        "ground_truth":    "Part 117 applies to all flightcrew members conducting passenger operations under Part 121. However, it does not apply to all-cargo operations conducted solely under Part 121 unless the certificate holder elects to apply Part 117. For cargo-only operations, certificate holders may use the older Part 121 Subpart Q rest requirements instead of Part 117, unless they opt into Part 117 for their cargo crews.",
        "relevant_doc_ids": ["cfr-046af492fb4a", "cfr-7d5ded315310"],
        "difficulty":      "hard",
        "notes":           "§ 117.1 scope exclusion for all-cargo",
    },
    {
        "query_id":        "CFR-012",
        "source":          "FAA_CFR",
        "query_type":      "edge_case",
        "question":        "Under 14 CFR Part 43, who is NOT authorized to perform preventive maintenance on an aircraft?",
        "ground_truth":    "Under § 43.3, preventive maintenance may only be performed by the certificated pilot who holds at least a private pilot certificate and is the registered owner of a non-commercial aircraft. A holder of a sport pilot certificate may only perform preventive maintenance on aircraft for which they hold a valid flight instructor certificate with sport pilot rating. Persons holding only a student pilot certificate or recreational pilot certificate are NOT authorized to perform preventive maintenance.",
        "relevant_doc_ids": ["cfr-cc8e13c82903"],
        "difficulty":      "hard",
        "notes":           "§ 43.3(g) preventive maintenance authorization",
    },
    {
        "query_id":        "CFR-013",
        "source":          "FAA_CFR",
        "query_type":      "comparative",
        "question":        "What is the difference between the applicability of 14 CFR Part 121 and Part 135 operating rules?",
        "ground_truth":    "Part 121 applies to air carriers and commercial operators conducting domestic, flag, and supplemental operations with large aircraft (generally more than 30 passenger seats or payloads exceeding 7,500 lbs) used in common carriage. Part 135 applies to commuter and on-demand operations — typically smaller aircraft (30 or fewer passenger seats, 7,500 lbs or less payload) used in air taxi and commuter services. Part 119 determines which part applies based on the type and size of operation.",
        "relevant_doc_ids": ["cfr-7d5ded315310", "cfr-8ebb5f5bcd9f", "cfr-21c4cca85996"],
        "difficulty":      "medium",
        "notes":           "§ 119.1 routing to Part 121 vs 135",
    },
    {
        "query_id":        "CFR-014",
        "source":          "FAA_CFR",
        "query_type":      "comparative",
        "question":        "How do the maintenance requirements differ between 14 CFR Part 145 repair stations and Part 43 individual certificated mechanics?",
        "ground_truth":    "Part 43 allows individual certificated mechanics (A&P certificate holders) to perform maintenance on aircraft they are authorized for. Part 145 governs certificated repair stations — organizations that employ mechanics and must have approved facilities, equipment, inspection systems, and a quality system. Repair stations may work on air carrier aircraft and must follow their Repair Station Manual and Operations Specifications. Individual mechanics under Part 43 work under personal certification authority; repair stations operate under organizational approval with broader capabilities and accountability.",
        "relevant_doc_ids": ["cfr-cc8e13c82903", "cfr-ebc2bc4aa1d2"],
        "difficulty":      "hard",
        "notes":           "Individual vs. organizational maintenance authority",
    },
    {
        "query_id":        "CFR-015",
        "source":          "FAA_CFR",
        "query_type":      "comparative",
        "question":        "How does the SMS framework in 14 CFR Part 5 differ from the safety requirements in Part 121 Subpart X?",
        "ground_truth":    "Part 5 establishes the overarching SMS framework applicable to Part 121 air carriers — requiring a systematic approach with four components: safety policy, safety risk management, safety assurance, and safety promotion. It mandates proactive hazard identification and risk management. Part 121 Subpart X (§ 121.433 onward) focuses on specific training and checking requirements for crewmembers. Part 5 is system-level organizational safety management; Part 121 Subpart X is crew qualification and proficiency. Part 5 was specifically designed to fulfill ICAO Annex 19 SMS standards at the certificate holder level.",
        "relevant_doc_ids": ["cfr-3a9bbbf95f1a", "cfr-7d5ded315310"],
        "difficulty":      "hard",
        "notes":           "SMS vs. training requirements distinction",
    },

    # =========================================================================
    # SOURCE: FAA_AD — 15 queries
    # =========================================================================

    {
        "query_id":        "AD-001",
        "source":          "FAA_AD",
        "query_type":      "factual",
        "question":        "What unsafe condition prompted AD 2023-08-04 for Boeing 787-8, 787-9, and 787-10 airplanes?",
        "ground_truth":    "AD 2023-08-04 was prompted by reports of loss of water pressure during flight and water leaks that affected multiple pieces of electronic equipment, caused by missing or incorrectly installed clamshell couplings in the door 1 and door 3 lavatory and galley potable water systems.",
        "relevant_doc_ids": ["ad-2d304546b5e4", "ad-fa65aa50e473"],
        "difficulty":      "easy",
        "notes":           "Direct AD content lookup",
    },
    {
        "query_id":        "AD-002",
        "source":          "FAA_AD",
        "query_type":      "factual",
        "question":        "What inspection does the FAA require for Boeing 787-9 and 787-10 airplanes regarding ram air turbine forward fittings?",
        "ground_truth":    "The FAA requires a high frequency eddy current (HFEC) or handheld X-ray fluorescence (XRF) spectrometer inspection to determine if the ram air turbine (RAT) forward fittings were manufactured with an incorrect titanium alloy material, based on multiple supplier notices of escapement (NOEs).",
        "relevant_doc_ids": ["ad-eea224e7d28e"],
        "difficulty":      "easy",
        "notes":           "Specific inspection method lookup",
    },
    {
        "query_id":        "AD-003",
        "source":          "FAA_AD",
        "query_type":      "factual",
        "question":        "What action does the FAA require for Boeing 737-9 airplanes following the in-flight departure of a mid-cabin door plug?",
        "ground_truth":    "The FAA adopted an AD that prohibits further flight of affected Boeing 737-9 airplanes until the airplane is inspected and all applicable corrective actions have been performed, following a report of an in-flight departure of a mid cabin door plug which resulted in rapid decompression.",
        "relevant_doc_ids": ["ad-8bcd767466a5"],
        "difficulty":      "easy",
        "notes":           "AD 2024-00993 — 737-9 door plug",
    },
    {
        "query_id":        "AD-004",
        "source":          "FAA_AD",
        "query_type":      "procedural",
        "question":        "What are the required actions under the FAA's AD for Boeing 787 airplanes regarding Captain's seat uncommanded forward movement?",
        "ground_truth":    "The AD requires inspections of affected Captain's and First Officer's seats for missing or cracked rocker switch caps and for cracked or nonfunctional switch cover assemblies, and applicable on-condition actions, following a report of uncommanded movement of the Captain's seat in the forward direction that caused a rapid descent.",
        "relevant_doc_ids": ["ad-750e5dc82994"],
        "difficulty":      "medium",
        "notes":           "Multi-step inspection requirement",
    },
    {
        "query_id":        "AD-005",
        "source":          "FAA_AD",
        "query_type":      "procedural",
        "question":        "What maintenance actions are required for Boeing 777 airplanes under the AD addressing wing anti-ice valve failure?",
        "ground_truth":    "The FAA AD for Boeing 777 airplanes with wing anti-ice (WAI) valve failure requires actions to address: (1) WAI valve failure that can result in undetected structural damage to leading edge slat assemblies, and (2) a failure of the autothrottle to disconnect after the pilot manually advanced the throttle. Required actions include inspections and modifications to both the WAI system and autothrottle system.",
        "relevant_doc_ids": ["ad-fb314ab4b9d7"],
        "difficulty":      "medium",
        "notes":           "Dual-issue AD",
    },
    {
        "query_id":        "AD-006",
        "source":          "FAA_AD",
        "query_type":      "procedural",
        "question":        "What actions are required for Airbus A320 series airplanes following the uncommanded pitch-down event?",
        "ground_truth":    "Following an uncommanded and limited pitch down event on an Airbus A320 where the autopilot remained engaged with a brief and limited loss of control, the FAA issued an emergency AD requiring operators to revise the existing airplane flight manual (AFM) — specifically providing instructions to pilots for autopilot disconnection and manual flight recovery procedures.",
        "relevant_doc_ids": ["ad-0f02cfd7a1a4"],
        "difficulty":      "medium",
        "notes":           "Emergency AD — AFM revision",
    },
    {
        "query_id":        "AD-007",
        "source":          "FAA_AD",
        "query_type":      "cross_ref",
        "question":        "How does AD 2024-01-01 relate to the subsequently issued AD for Boeing 787 faucet control module leaks?",
        "ground_truth":    "AD 2024-01-01 required repetitive general visual inspections of the area under all lavatory washbasins for evidence of intermittent and active leaks at the faucet control module (FCM). A subsequent AD superseded AD 2024-01-01 because Boeing determined that the FCM had a design issue (not just installation), requiring additional actions beyond inspection — specifically, replacement of affected FCMs with improved design parts. The superseding AD retained the inspection requirement while adding the FCM replacement action.",
        "relevant_doc_ids": ["ad-a3c81862018b", "ad-50dcbcc28a14"],
        "difficulty":      "hard",
        "notes":           "AD supersession chain",
    },
    {
        "query_id":        "AD-008",
        "source":          "FAA_AD",
        "query_type":      "cross_ref",
        "question":        "What is the relationship between AD 2022-15-06 and its superseding AD for Boeing 777 transorb modules?",
        "ground_truth":    "AD 2022-15-06 required disconnecting certain connectors and capping and stowing the wires that had been attached to affected transorb modules on Boeing 777 airplanes. The superseding AD expanded the scope because the FAA determined that additional connectors beyond those identified in AD 2022-15-06 were also affected. The superseding AD also introduced a replacement action (installing new parts) as a terminating action to the repetitive disconnection requirement.",
        "relevant_doc_ids": ["ad-2440afab3067", "ad-9808e686708d"],
        "difficulty":      "hard",
        "notes":           "Expanding applicability in supersession",
    },
    {
        "query_id":        "AD-009",
        "source":          "FAA_AD",
        "query_type":      "edge_case",
        "question":        "What triggered the FAA to withdraw its NPRM for Boeing 787 door assist handles, and what does this indicate about the AD process?",
        "ground_truth":    "The FAA withdrew an NPRM that proposed to address Boeing 787 door assist handles pulling loose from their lower attachment point in the doorway support bracket during pre-flight checks. The withdrawal indicates that either a revised NPRM was issued to address updated information, or the safety issue was resolved through other means (such as a manufacturer service bulletin incorporated into a different AD). The AD process allows NPRMs to be withdrawn and reissued when new data emerges.",
        "relevant_doc_ids": ["ad-c8bb291afe6d"],
        "difficulty":      "hard",
        "notes":           "NPRM withdrawal process",
    },
    {
        "query_id":        "AD-010",
        "source":          "FAA_AD",
        "query_type":      "edge_case",
        "question":        "For Boeing 737-800 airplanes, what is the special compliance time condition in the AD addressing skin under the drag link assembly?",
        "ground_truth":    "The AD addressing the Boeing 737-800 drag link assembly skin determined that the compliance time for the initial ultrasonic inspection required by AD 2019-11-06 is insufficient for certain airplanes. The superseding AD requires reducing the compliance time for the ultrasonic inspection of the skin under the drag link assembly — meaning certain airplanes must complete the inspection sooner than originally required, based on revised fatigue analysis.",
        "relevant_doc_ids": ["ad-85e160ffe847"],
        "difficulty":      "hard",
        "notes":           "Reduced compliance time — edge case in AD revision",
    },
    {
        "query_id":        "AD-011",
        "source":          "FAA_AD",
        "query_type":      "edge_case",
        "question":        "What manufacturing defect prompted an AD for Airbus A320 family airplanes involving cold working process deviations?",
        "ground_truth":    "A review of the cold working process on the Airbus A320 family assembly line detected a deviation in the manufacturing process, resulting in fastener holes in certain center fuselage frames not receiving the proper cold working treatment. The AD requires repetitive inspections of the nominal design condition of the fastener holes to detect fatigue cracking that could result from this manufacturing deficiency.",
        "relevant_doc_ids": ["ad-8de4a5ed0b5b"],
        "difficulty":      "medium",
        "notes":           "Manufacturing process deviation AD",
    },
    {
        "query_id":        "AD-012",
        "source":          "FAA_AD",
        "query_type":      "comparative",
        "question":        "What is the difference between an FAA Final Rule AD and a Notice of Proposed Rulemaking (NPRM) AD in the Federal Register?",
        "ground_truth":    "A Final Rule AD (indicated by 'The FAA is adopting' or 'The FAA is superseding') is immediately effective law that operators must comply with within the specified compliance time. An NPRM AD (indicated by 'The FAA proposes to adopt') is a proposed rule open for public comment — operators are not yet required to comply, but the proposed actions indicate what will likely become mandatory. An NPRM becomes a Final Rule after the comment period closes and the FAA addresses comments. Some urgent safety situations result in an Emergency AD issued as a Final Rule without prior NPRM.",
        "relevant_doc_ids": ["ad-2d304546b5e4", "ad-ef13ee2536d2"],
        "difficulty":      "medium",
        "notes":           "Understanding AD legal status",
    },
    {
        "query_id":        "AD-013",
        "source":          "FAA_AD",
        "query_type":      "comparative",
        "question":        "How do ADs for Boeing 787 structural issues differ from ADs addressing avionics/systems issues in terms of required actions?",
        "ground_truth":    "ADs for Boeing 787 structural issues (such as forward pressure bulkhead gaps, SOB splice plate preload, titanium alloy material non-conformances) typically require one-time or repetitive inspections (visual, ultrasonic, eddy current, or X-ray fluorescence) with on-condition repair or replacement. ADs for avionics/systems issues (such as transponder MOPS failures, uncommanded MCP altitude changes, radio altimeter NCD outputs) typically require software updates, hardware replacements, or AFM revisions — often as one-time actions with no repetitive inspection component.",
        "relevant_doc_ids": ["ad-6fd835288283", "ad-281b05370620"],
        "difficulty":      "hard",
        "notes":           "Structural vs. avionics AD action types",
    },
    {
        "query_id":        "AD-014",
        "source":          "FAA_AD",
        "query_type":      "comparative",
        "question":        "What is the difference between how the FAA addresses engine turbine disk material defects versus fuselage structural cracks in ADs?",
        "ground_truth":    "For engine turbine disk material defects (such as HPT disks manufactured from powder metal with iron inclusion in GE90/GEnx engines, or incorrect titanium in Pratt & Whitney engines), ADs typically require immediate replacement before exceeding a reduced life limit — since material flaws can cause uncontained failures without visible precursors. For fuselage structural cracks (such as 737 chem-mill skin cracks, 777 fuselage skin at underwing longeron), ADs require repetitive inspections at specified intervals since crack propagation is detectable and manageable over time.",
        "relevant_doc_ids": ["ad-df215fc594cf", "ad-8a3f9c476322"],
        "difficulty":      "hard",
        "notes":           "Damage tolerance vs. retirement-for-cause approach",
    },
    {
        "query_id":        "AD-015",
        "source":          "FAA_AD",
        "query_type":      "factual",
        "question":        "What safety concern prompted the FAA to issue an AD for Boeing 737 MAX airplanes regarding in-flight cabin temperatures?",
        "ground_truth":    "The AD was prompted by reports of in-flight events of excessive cabin and flight deck temperatures that could not be controlled by the flightcrew using existing procedures. The AD requires revising the existing airplane flight manual (AFM) to provide the flightcrew with operating procedures (non-normal) for managing excessive temperatures.",
        "relevant_doc_ids": ["ad-5904275c03ce"],
        "difficulty":      "easy",
        "notes":           "737 MAX temperature control AD",
    },

    # =========================================================================
    # SOURCE: FAA_AC — 15 queries
    # =========================================================================

    {
        "query_id":        "AC-001",
        "source":          "FAA_AC",
        "query_type":      "factual",
        "question":        "What is the purpose of FAA AC 120-76D regarding Electronic Flight Bags?",
        "ground_truth":    "AC 120-76D provides guidance on the operational use of Electronic Flight Bags (EFBs). It is intended for all operators conducting flight operations under 14 CFR parts 91K, 121, 125, and 135, and provides an acceptable means (but not the only means) for authorization of EFB use — covering EFB types, hardware requirements, software applications, human factors considerations, and operational approval procedures.",
        "relevant_doc_ids": ["ac-61b44f8942f4"],
        "difficulty":      "easy",
        "notes":           "AC purpose statement",
    },
    {
        "query_id":        "AC-002",
        "source":          "FAA_AC",
        "query_type":      "factual",
        "question":        "According to FAA AC 120-42B, what is ETOPS and under which 14 CFR section is ETOPS authorization obtained?",
        "ground_truth":    "ETOPS stands for Extended Operations — authorization for twin-engine aircraft to fly routes that take them more than 60 minutes from a diversion airport at one-engine-inoperative cruise speed. ETOPS authorization for part 121 certificate holders is obtained under 14 CFR § 121.161. The AC provides guidance for obtaining operational approval for ETOPS including maintenance requirements, training, and flight operations procedures.",
        "relevant_doc_ids": ["ac-46192095ce2b"],
        "difficulty":      "easy",
        "notes":           "ETOPS definition and regulatory basis",
    },
    {
        "query_id":        "AC-003",
        "source":          "FAA_AC",
        "query_type":      "factual",
        "question":        "What are the four SMS components required under FAA AC 120-92B for aviation service providers?",
        "ground_truth":    "According to AC 120-92B (implementing 14 CFR Part 5), the four required components of a Safety Management System are: (1) Safety Policy and Objectives — including management commitment, safety accountabilities, and SMS documentation; (2) Safety Risk Management — hazard identification and risk assessment/mitigation; (3) Safety Assurance — safety performance monitoring and management of change; and (4) Safety Promotion — training, communication, and safety culture.",
        "relevant_doc_ids": ["ac-51b8d0ac6a94"],
        "difficulty":      "easy",
        "notes":           "ICAO Annex 19 four-component SMS",
    },
    {
        "query_id":        "AC-004",
        "source":          "FAA_AC",
        "query_type":      "procedural",
        "question":        "What steps must an air carrier take to obtain ETOPS authorization under FAA AC 120-42B?",
        "ground_truth":    "Under AC 120-42B, an air carrier seeking ETOPS authorization must: (1) Meet ETOPS in-service experience requirements — demonstrating operational reliability with the specific airframe-engine combination; (2) Establish an ETOPS maintenance program including engine condition monitoring, propulsion system reliability, and ETOPS significant system maintenance tasks; (3) Develop ETOPS flight operations procedures including diversion planning, fuel planning for ETOPS alternates, and crew training; (4) Submit application to the FAA for ETOPS authority, including operations specifications amendments; (5) Maintain operational reliability above the required threshold (target: 0.05 IFSD rate per 1,000 hours for 180-minute ETOPS).",
        "relevant_doc_ids": ["ac-46192095ce2b"],
        "difficulty":      "hard",
        "notes":           "Multi-step ETOPS approval process",
    },
    {
        "query_id":        "AC-005",
        "source":          "FAA_AC",
        "query_type":      "procedural",
        "question":        "According to FAA AC 120-51E, what are the key components of a CRM training program for flight crews?",
        "ground_truth":    "AC 120-51E identifies that a comprehensive CRM training program should include: (1) Initial CRM training providing awareness of CRM concepts and skills; (2) Recurrent CRM training reinforcing skills and addressing new information; (3) Integration of CRM into all flight training (not as standalone module); (4) LOFT (Line-Oriented Flight Training) incorporating CRM scenarios; (5) Assessment of CRM behaviors during training and checking. Core CRM topics include communications processes, situational awareness, problem solving and decision making, workload management, and teamwork.",
        "relevant_doc_ids": ["ac-e5c277cf1dd9"],
        "difficulty":      "medium",
        "notes":           "CRM training program structure",
    },
    {
        "query_id":        "AC-006",
        "source":          "FAA_AC",
        "query_type":      "procedural",
        "question":        "What does FAA AC 39-7D require of aircraft owners and operators for compliance with Airworthiness Directives?",
        "ground_truth":    "AC 39-7D provides that aircraft owners/operators must: (1) Determine if any ADs are applicable to their aircraft, engine, propeller, or appliances; (2) Comply with each applicable AD within the specified compliance time; (3) Record AD compliance in the aircraft maintenance records per § 43.9, including the AD number, revision date, method of compliance, and date of compliance; (4) For repetitive ADs, track and record each recurrence; (5) For ADs with Alternative Methods of Compliance (AMOC), obtain FAA approval before using the alternative. Non-compliance with an applicable AD is a violation making the aircraft unairworthy.",
        "relevant_doc_ids": ["ac-6f76409314ac"],
        "difficulty":      "medium",
        "notes":           "AD compliance obligations",
    },
    {
        "query_id":        "AC-007",
        "source":          "FAA_AC",
        "query_type":      "cross_ref",
        "question":        "How does FAA AC 120-92B on SMS relate to the requirements in 14 CFR Part 5?",
        "ground_truth":    "AC 120-92B provides guidance specifically for implementing the SMS requirements mandated in 14 CFR Part 5 for Part 121 air carriers. Part 5 establishes the regulatory requirement for SMS (safety policy, safety risk management, safety assurance, safety promotion); AC 120-92B explains how to implement each component in practice. The AC is not mandatory itself — it is an acceptable means of compliance. Carriers may use alternative methods, but the AC represents FAA's accepted interpretation of how Part 5 requirements should be fulfilled.",
        "relevant_doc_ids": ["ac-51b8d0ac6a94", "cfr-3a9bbbf95f1a"],
        "difficulty":      "medium",
        "notes":           "AC vs. regulation relationship",
    },
    {
        "query_id":        "AC-008",
        "source":          "FAA_AC",
        "query_type":      "cross_ref",
        "question":        "What is the relationship between FAA AC 120-109A on stall training and the regulatory requirements in 14 CFR Part 121?",
        "ground_truth":    "AC 120-109A provides guidance for implementing stall prevention and recovery training that is required by 14 CFR Part 121. Specifically, § 121.423 requires extended envelope training (including stalls) for Part 121 air carriers, and § 121.424 specifies initial and recurrent training requirements. AC 120-109A provides acceptable means of compliance for designing, conducting, and evaluating stall training, including the FAA-endorsed stall recovery template developed with Airbus, Boeing, Bombardier, and other manufacturers.",
        "relevant_doc_ids": ["ac-e5c277cf1dd9"],
        "difficulty":      "medium",
        "notes":           "Training AC to regulatory requirement linkage",
    },
    {
        "query_id":        "AC-009",
        "source":          "FAA_AC",
        "query_type":      "edge_case",
        "question":        "Under FAA AC 120-118, what are the special conditions for authorization of CAT III autoland operations?",
        "ground_truth":    "AC 120-118 provides that CAT III authorization requires the most stringent conditions. CAT III operations (Decision Height below 50 ft or no DH for CAT IIIc) require: aircraft certification for CAT III (including redundant autoland systems and fail-operational capability), airport certification with CAT III ILS, operator-specific training and checking for all crewmembers, specific maintenance programs for CAT III critical systems, approved CAT III Operations Specifications, and demonstrated operational capability. Runway visual range (RVR) minimums vary by CAT IIIa (RVR 600 ft minimum), IIIb (RVR 150-600 ft), and IIIc (no visibility requirement).",
        "relevant_doc_ids": ["ac-e5c277cf1dd9"],
        "difficulty":      "hard",
        "notes":           "CAT III special conditions",
    },
    {
        "query_id":        "AC-010",
        "source":          "FAA_AC",
        "query_type":      "edge_case",
        "question":        "What exception does FAA AC 91-67 provide for operating aircraft with inoperative equipment under 14 CFR Part 91?",
        "ground_truth":    "AC 91-67 explains that under § 91.213, a Part 91 operator may legally fly with certain inoperative equipment if: (1) the aircraft has a FAA-approved Minimum Equipment List (MEL), the inoperative item is listed as deferrable, and the item is properly deactivated and placarded; OR (2) without an MEL, the equipment is not required by the airworthiness standards, not required for the specific operation, is removed and placarded or deactivated, and the FAA-approved flight manual does not prohibit operation without it. This provides a legal mechanism to dispatch aircraft with known equipment faults without grounding the aircraft.",
        "relevant_doc_ids": ["ac-3ed8e29db9d8"],
        "difficulty":      "hard",
        "notes":           "MEL vs. § 91.213(d) dispatch options",
    },
    {
        "query_id":        "AC-011",
        "source":          "FAA_AC",
        "query_type":      "edge_case",
        "question":        "According to FAA AC 117-1, what specific type of rest facility is required for augmented crew rest during long-range operations?",
        "ground_truth":    "AC 117-1 provides that for augmented crew rest during long-range Part 121 operations, the rest facility must provide the ability for the resting crewmember to sleep in a bunk (not a seat). The facility must allow control of light and temperature, provide adequate acoustical separation from the flight deck and passenger cabin, and must meet one of three rest facility classes (Class 1: bunk equivalent to lower berth; Class 2: flat surface; Class 3: seat in non-overhead area) as defined in 14 CFR Part 25 aircraft certification standards (§ 25.789).",
        "relevant_doc_ids": ["ac-c5f0ef21650f"],
        "difficulty":      "hard",
        "notes":           "Augmented crew rest facility classes",
    },
    {
        "query_id":        "AC-012",
        "source":          "FAA_AC",
        "query_type":      "comparative",
        "question":        "How does FAA AC 43.13-1B differ from AC 43.13-2B in scope and application?",
        "ground_truth":    "AC 43.13-1B covers acceptable methods, techniques, and practices for aircraft inspection and repair of non-pressurized areas of civil aircraft when no manufacturer repair or maintenance instructions exist. AC 43.13-2B covers acceptable methods for aircraft alterations — specifically for the inspection and alteration of non-pressurized areas of civil aircraft of 12,500 lbs gross weight or less. The key differences: 1B is for repairs/maintenance, 2B is for alterations; 1B applies to all civil aircraft, 2B is limited to aircraft ≤12,500 lbs; both serve as references under § 43.13 when manufacturer data is unavailable.",
        "relevant_doc_ids": ["ac-7a3a3fb649de", "ac-526f8e8f5644"],
        "difficulty":      "medium",
        "notes":           "Repair vs. alteration guidance distinction",
    },
    {
        "query_id":        "AC-013",
        "source":          "FAA_AC",
        "query_type":      "comparative",
        "question":        "What is the difference between ETOPS authorization and Polar Operations authorization under FAA AC 120-42B?",
        "ground_truth":    "Under AC 120-42B: ETOPS authorization addresses extended operations over oceanic and remote areas where the diversion time to an adequate airport exceeds 60 minutes at single-engine cruise speed — focused on engine reliability, ETOPS alternate airports, and fuel planning for single-engine diversion. Polar operations authorization addresses flights in polar regions (above 78°N or below 60°S latitude), focusing on additional challenges including magnetic compass unreliability, communication blackouts requiring satellite communications, polar weather conditions, and limited diversion airports in polar regions. An operator may hold ETOPS authority without polar authority and vice versa.",
        "relevant_doc_ids": ["ac-46192095ce2b"],
        "difficulty":      "hard",
        "notes":           "ETOPS vs polar operations distinction",
    },
    {
        "query_id":        "AC-014",
        "source":          "FAA_AC",
        "query_type":      "comparative",
        "question":        "How does FAA AC 120-71B's guidance on Standard Operating Procedures differ for Part 121 versus Part 91 operators?",
        "ground_truth":    "AC 120-71B focuses primarily on Part 121 and Part 135 operators who have regulatory requirements for operations manuals (§ 121.133, § 135.83). For Part 121 operators, SOPs are operationally mandatory and must be trained, evaluated in checking events, and included in the Operations Manual. The AC covers SOP design, pilot monitoring duties, and checklist philosophy as part of formal training programs. For Part 91 operators, SOPs are best practices — not regulatory mandates for most operations — though Part 91K (fractional ownership) operators have SOP requirements similar to Part 121. The AC's human factors guidance applies broadly, but the regulatory enforcement differs.",
        "relevant_doc_ids": ["ac-d2d482883aee"],
        "difficulty":      "hard",
        "notes":           "Regulatory vs. best-practice SOP applicability",
    },
    {
        "query_id":        "AC-015",
        "source":          "FAA_AC",
        "query_type":      "factual",
        "question":        "According to FAA AC 120-85A, what is the primary focus of guidance on air cargo operations?",
        "ground_truth":    "AC 120-85A focuses on cargo loading guidance for air operators and is intended for air operators, OEMs, STC holders, PMA holders, TSO holders, and aircraft owners and operators. The primary focus is on cargo loading — including weight and balance, tie-down requirements, cargo fire containment, hazardous materials handling, and compliance with 14 CFR § 121.583 (cargo operations requirements) and § 91.9 (weight limits). It covers both structural and fire safety aspects of cargo operations.",
        "relevant_doc_ids": ["ac-53cdf4cb9ad3"],
        "difficulty":      "easy",
        "notes":           "AC 120-85A scope",
    },

    # =========================================================================
    # SOURCE: DGCA_CAR — 15 queries
    # =========================================================================

    {
        "query_id":        "DGCA-001",
        "source":          "DGCA_CAR",
        "query_type":      "factual",
        "question":        "Under DGCA CAR Section 7 Series J Part III, what is the maximum flight time in 7 consecutive days for flight crew in scheduled air transport operations?",
        "ground_truth":    "Under DGCA CAR Section 7 Series J Part III (Rev 1, 8 Jan 2024), the maximum cumulative flight time in 7 consecutive days is 35 hours, and the maximum cumulative duty period in 7 consecutive days is 60 hours.",
        "relevant_doc_ids": ["dgca-2a7204c2bd17"],
        "difficulty":      "easy",
        "notes":           "Table at Para 8.1 — direct lookup",
    },
    {
        "query_id":        "DGCA-002",
        "source":          "DGCA_CAR",
        "query_type":      "factual",
        "question":        "What is the maximum cumulative flight time in 365 consecutive days under DGCA FDTL regulations for scheduled air transport?",
        "ground_truth":    "Under DGCA CAR Section 7 Series J Part III, the maximum cumulative flight time in 365 consecutive days is 1,000 hours, with a maximum cumulative duty period of 1,800 hours over the same period.",
        "relevant_doc_ids": ["dgca-2a7204c2bd17"],
        "difficulty":      "easy",
        "notes":           "Table at Para 8.5 — annual limits",
    },
    {
        "query_id":        "DGCA-003",
        "source":          "DGCA_CAR",
        "query_type":      "factual",
        "question":        "Under DGCA CAR Section 5 Series F Part III, what is the regulatory basis for conducting breath-analyzer examinations on flight crew?",
        "ground_truth":    "The breath-analyzer examination procedure is issued under the provisions of Rule 24 read with Rule 133A of the Aircraft Rules, 1937. Rule 24 prohibits any person from acting as a flight crew member while having alcohol in their blood. The CAR provides that no flight crew member shall exercise privileges of their license with a blood alcohol level exceeding the prescribed limit, and specifies pre-flight and post-flight breath-analyzer testing procedures.",
        "relevant_doc_ids": ["dgca-e9bd251ec799"],
        "difficulty":      "easy",
        "notes":           "Legal basis for alcohol testing",
    },
    {
        "query_id":        "DGCA-004",
        "source":          "DGCA_CAR",
        "query_type":      "procedural",
        "question":        "What is the minimum rest period required for flight crew crossing more than 7 time zones under DGCA CAR Section 7 Series J Part III?",
        "ground_truth":    "Under DGCA CAR Section 7 Series J Part III, Para 10.1, the minimum rest before undertaking a flight duty period must be at least as long as the preceding duty period, OR the greater of: (i) 12 hours for standard operations, (ii) 18 hours for crossing more than 3 up to 7 time zones, or (iii) 36 hours for crossing more than 7 time zones. For more than 7 time zone crossings, the minimum rest is 36 hours.",
        "relevant_doc_ids": ["dgca-2a7204c2bd17"],
        "difficulty":      "medium",
        "notes":           "Para 10.1 time zone rest requirements",
    },
    {
        "query_id":        "DGCA-005",
        "source":          "DGCA_CAR",
        "query_type":      "procedural",
        "question":        "Under DGCA CAR Section 7 Series J Part III, what are the rules for split duty and how can the Flight Duty Period be extended?",
        "ground_truth":    "Under Para 9 of DGCA CAR Section 7 Series J Part III, split duty extension rules are: (1) A break of less than 3 hours — no FDP extension permitted; (2) A break between 3 and 10 hours — FDP can be extended by a period equal to half the consecutive hours of break taken; (3) A break of more than 10 hours — no extension permitted. Additional conditions: parts of the FDP before and after the break shall not each exceed 10 hours; split duty counts in full as FDP; no extension is permitted if the FDP encroaches on the night duty period; if the break exceeds 6 consecutive hours or encroaches on WOCL, suitable accommodation must be provided.",
        "relevant_doc_ids": ["dgca-2a7204c2bd17"],
        "difficulty":      "medium",
        "notes":           "Split duty table Para 9",
    },
    {
        "query_id":        "DGCA-006",
        "source":          "DGCA_CAR",
        "query_type":      "procedural",
        "question":        "According to DGCA CAR Section 3 Series C Part I on monsoon operations, what pre-flight crew briefing requirements apply to monsoon season flights?",
        "ground_truth":    "Under DGCA Operations Circular 04 of 2023 on Monsoon Operations, pre-flight crew briefing (whether self-briefed or by a dispatcher) must cover: (1) Aircraft status, especially MEL items and their impact on monsoon operations; (2) Latest departure, enroute, destination, and alternate weather; (3) Changes to enroute, destination and alternate weather must be transmitted to crew whenever there is significant deterioration post pre-flight briefing; (4) Fuel uplift calculation must be done judiciously considering monsoon weather deviations; (5) Crew should keep themselves updated with required weather information from all available sources.",
        "relevant_doc_ids": ["dgca-8e19b561239d"],
        "difficulty":      "medium",
        "notes":           "Section 4.1 pre-flight requirements",
    },
    {
        "query_id":        "DGCA-007",
        "source":          "DGCA_CAR",
        "query_type":      "cross_ref",
        "question":        "How does DGCA CAR Section 7 Series J Part III on flight crew FDTL relate to ICAO Annex 6 fatigue management requirements?",
        "ground_truth":    "DGCA CAR Section 7 Series J Part III is issued to implement ICAO Annex 6, Part 1 requirements. ICAO Annex 6 requires each State to establish regulations for managing fatigue — prescriptive Flight Time, Flight Duty Period, Duty Period, and Rest Period limitations, and optionally FRMS regulations. DGCA CAR Section 7 Series J Part III is India's prescriptive fatigue management regulation, issued under Rule 42A and Rule 133A of Aircraft Rules 1937, that fulfills this ICAO obligation. Operators subject to this CAR must establish their own FDTL scheme within these regulatory limits.",
        "relevant_doc_ids": ["dgca-2a7204c2bd17"],
        "difficulty":      "medium",
        "notes":           "ICAO-DGCA regulatory chain",
    },
    {
        "query_id":        "DGCA-008",
        "source":          "DGCA_CAR",
        "query_type":      "cross_ref",
        "question":        "What is the relationship between DGCA CAR-147 and CAR-66 in the Indian aviation maintenance regulatory framework?",
        "ground_truth":    "CAR-147 (Maintenance Training Organisation requirements) was developed subsequent to and dependent upon CAR-66 (Aircraft Maintenance Licence requirements). CAR-66 defines the knowledge and examination standards that maintenance engineers must meet to obtain an Aircraft Maintenance Licence in India. CAR-147 specifies the requirements for organisations seeking approval to conduct the basic maintenance training and examinations as specified in CAR-66. CAR-147 is based on EASA Part 147, and training organisations approved under CAR-147 may conduct the training and (subject to DGCA permission) examinations required for CAR-66 licences.",
        "relevant_doc_ids": ["dgca-8920fc7f65a1"],
        "difficulty":      "hard",
        "notes":           "CAR-147 foreword explicitly references CAR-66",
    },
    {
        "query_id":        "DGCA-009",
        "source":          "DGCA_CAR",
        "query_type":      "edge_case",
        "question":        "Under DGCA CAR Section 7 Series J Part III, what is the definition and acclimatization rule for a crew member crossing time zones?",
        "ground_truth":    "Under Para 3.1 of DGCA CAR Section 7 Series J Part III, 'Acclimatized' means a state in which a crew member's circadian biological clock is synchronized to the time zone where they are located. A crew member is considered acclimatized to a 3-hour wide time zone surrounding the local time at the point of departure. When the local time at the flight duty commencement point differs by more than 3 hours from the local time at the next duty start point, the crew member is considered acclimatized to the departure time zone for the first 48 hours. After 48 hours, they are considered acclimatized to the local time where they start their next duty.",
        "relevant_doc_ids": ["dgca-2a7204c2bd17"],
        "difficulty":      "hard",
        "notes":           "Acclimatization definition — nuanced 48-hour rule",
    },
    {
        "query_id":        "DGCA-010",
        "source":          "DGCA_CAR",
        "query_type":      "edge_case",
        "question":        "Under DGCA CAR Section 3 Series M Part VI, when can an operator refuse carriage to an unruly passenger prior to boarding?",
        "ground_truth":    "Under DGCA CAR Section 3 Series M Part VI (Handling of Unruly Passengers), Para 4.4 provides that passengers who are likely to be unruly must be carefully monitored, and if their behaviour prior to boarding is such that there are reasonable grounds to suspect they would be disruptive during the flight, the operator may refuse to carry them. The CAR requires operators to establish conditions of carriage with statutory warnings about unruly passenger offences, and SOPs including crew roles when dealing with potentially disruptive passengers before and during flight.",
        "relevant_doc_ids": ["dgca-8b3acd3b41f5"],
        "difficulty":      "medium",
        "notes":           "Pre-boarding refusal of carriage",
    },
    {
        "query_id":        "DGCA-011",
        "source":          "DGCA_CAR",
        "query_type":      "edge_case",
        "question":        "Under DGCA CAR-147, can an existing training organisation approved under the older CAR Section 2 Series E be considered for CAR-147 approval?",
        "ground_truth":    "Yes. Under DGCA CAR-147, Para 147.A.01(b), existing training organisations approved under CAR Section 2 Series E may be considered for approval under CAR-147, provided they make a long-term viable agreement and technical arrangement with an approved maintenance organisation for imparting practical training to cover the scope of approval. This transition provision allowed older approved organisations to migrate to the harmonized CAR-147 standard (effective 1 March 2018).",
        "relevant_doc_ids": ["dgca-8920fc7f65a1"],
        "difficulty":      "hard",
        "notes":           "Transition provision in 147.A.01(b)",
    },
    {
        "query_id":        "DGCA-012",
        "source":          "DGCA_CAR",
        "query_type":      "comparative",
        "question":        "How do the FDTL limits for cabin crew under DGCA CAR Section 7 Series J Part I differ from flight crew limits under Series J Part III?",
        "ground_truth":    "Both DGCA CARs are based on ICAO Annex 6 and share the same overall structure, but cabin crew (Series J Part I) and flight crew (Series J Part III) have different regulatory foundations: Cabin crew FDTL under Series J Part I is issued for crew who perform passenger safety duties but are not flight crew members. Flight crew FDTL under Series J Part III applies specifically to pilots engaged in scheduled air transport. The cumulative flight time limits (e.g., 1,000 hours in 365 days) apply similarly to both, but operational definitions, rest facility classes, and augmented crew provisions may differ — flight crew have more stringent FDP calculation rules due to actual aircraft control responsibilities.",
        "relevant_doc_ids": ["dgca-fd842b9dea24", "dgca-2a7204c2bd17"],
        "difficulty":      "hard",
        "notes":           "Cabin vs. flight crew FDTL comparison",
    },
    {
        "query_id":        "DGCA-013",
        "source":          "DGCA_CAR",
        "query_type":      "comparative",
        "question":        "How do DGCA FDTL regulations compare to 14 CFR Part 117 in terms of annual flight time limits?",
        "ground_truth":    "Both DGCA CAR Section 7 Series J Part III and 14 CFR Part 117 set an annual flight time limit of 1,000 hours in any 365 consecutive days for flight crew in scheduled passenger operations. This alignment reflects both regulations being derived from ICAO Annex 6 standards. However, differences exist in other limits: Part 117 uses different monthly and weekly limits (100 hours/month, 1,000 hours/year) while DGCA uses 7-day (35 hours), 14-day (65 hours), 28-day (100 hours), and 90-day (300 hours) rolling windows. The DGCA framework provides more granular short-term limits.",
        "relevant_doc_ids": ["dgca-2a7204c2bd17", "cfr-046af492fb4a"],
        "difficulty":      "hard",
        "notes":           "India vs. US FDTL framework comparison",
    },
    {
        "query_id":        "DGCA-014",
        "source":          "DGCA_CAR",
        "query_type":      "factual",
        "question":        "What is the scope of DGCA CAR-147 in terms of which organizations it applies to?",
        "ground_truth":    "DGCA CAR-147 (Basic) is applicable to: (a) Approved aircraft/engine manufacturing and maintenance organisations registered in India that intend to impart basic aircraft maintenance training on aircraft, powerplant, and their systems; and (b) The section establishes requirements for organisations seeking approval to conduct Basic Maintenance Training as specified in CAR-66 and in accordance with Rule 133B of the Aircraft Rules, 1937. The CAR specifies conditions for issue, renewal, suspension, and revocation of certificates attached to the approval.",
        "relevant_doc_ids": ["dgca-8920fc7f65a1"],
        "difficulty":      "easy",
        "notes":           "147.A.01 applicability",
    },
    {
        "query_id":        "DGCA-015",
        "source":          "DGCA_CAR",
        "query_type":      "procedural",
        "question":        "Under DGCA CAR Section 5 Series F Part III, what is the procedure for pre-flight breath-analyzer examination of flight crew?",
        "ground_truth":    "Under DGCA CAR Section 5 Series F Part III, the pre-flight breath-analyzer examination procedure requires: (1) Testing must be conducted before each flight duty; (2) The test is conducted using calibrated breath-analyzer equipment; (3) Results showing any detectable alcohol result in the crew member being declared unfit for duty; (4) The CAR emphasizes that even at zero blood alcohol levels, hangover effects from congeners can last 15-18 hours and impair performance for up to 36 hours — the safety standard is zero tolerance during flight duties; (5) Equipment must be regularly calibrated by approved calibration agencies.",
        "relevant_doc_ids": ["dgca-e9bd251ec799"],
        "difficulty":      "medium",
        "notes":           "Pre-flight alcohol testing procedure",
    },

    # =========================================================================
    # SOURCE: SKYBRARY — 15 queries
    # =========================================================================

    {
        "query_id":        "SKY-001",
        "source":          "SKYBRARY",
        "query_type":      "factual",
        "question":        "According to SKYbrary, what percentage of unstabilised approaches result in a go-around being executed?",
        "ground_truth":    "According to SKYbrary's article on Go-Around Decision Making, only 3% of unstabilised approaches result in a go-around being flown. Conversely, 97% of unstabilised approaches continue to a landing contrary to airline SOPs. Between 3 and 4% of all approaches are reported or recorded as unstabilised.",
        "relevant_doc_ids": ["sky-bdbf1f6e7628"],
        "difficulty":      "easy",
        "notes":           "Key statistic from go-around article",
    },
    {
        "query_id":        "SKY-002",
        "source":          "SKYBRARY",
        "query_type":      "factual",
        "question":        "According to SKYbrary, what are the standard stabilised approach gate heights for IMC and VMC conditions?",
        "ground_truth":    "According to SKYbrary's article on Stabilised Approach, the standard stabilised approach gates are: 1,000 ft above the Touchdown Zone Elevation (TDZE) in IMC (Instrument Meteorological Conditions), and 500 ft above TDZE in VMC (Visual Meteorological Conditions). Some operators use higher gates. The approach must be abandoned and a go-around executed if stabilised criteria have not been achieved by these gates.",
        "relevant_doc_ids": ["sky-2a3446827ff9"],
        "difficulty":      "easy",
        "notes":           "Stabilised approach gates — factual",
    },
    {
        "query_id":        "SKY-003",
        "source":          "SKYBRARY",
        "query_type":      "factual",
        "question":        "What are the four components of the ICAO SMS framework as described in SKYbrary?",
        "ground_truth":    "According to SKYbrary's SMS article, the four components of the ICAO SMS framework (as required under ICAO Annex 19) are: (1) Safety Policy and Objectives — management commitment, safety accountabilities, key safety personnel, SMS documentation; (2) Safety Risk Management — hazard identification, safety risk assessment and mitigation, safety management of change; (3) Safety Assurance — safety performance monitoring, management of change, continuous improvement; and (4) Safety Promotion — training, communication, and safety culture.",
        "relevant_doc_ids": ["sky-493df911dd43"],
        "difficulty":      "easy",
        "notes":           "ICAO four-pillar SMS structure",
    },
    {
        "query_id":        "SKY-004",
        "source":          "SKYBRARY",
        "query_type":      "procedural",
        "question":        "According to SKYbrary, what are the immediate actions a flight crew should take when initiating an emergency descent due to loss of pressurisation?",
        "ground_truth":    "According to SKYbrary's Emergency Descent article, the immediate actions are: (1) Don oxygen masks immediately (memory item); (2) Declare an emergency with ATC — 'MAYDAY, MAYDAY, MAYDAY, [callsign], emergency descent'; (3) Set transponder 7700; (4) Begin descent — typically at MMo/VMo and maximum authorized thrust reduction; (5) Target 10,000 feet (or MEA if higher); (6) Brief cabin crew — activate seatbelt signs; (7) Complete QRH checklist.",
        "relevant_doc_ids": ["sky-fe966cb6b396"],
        "difficulty":      "medium",
        "notes":           "Emergency descent procedure steps",
    },
    {
        "query_id":        "SKY-005",
        "source":          "SKYBRARY",
        "query_type":      "procedural",
        "question":        "What decision must a flight crew make at V1 during engine failure on takeoff, and what are the procedures for each choice?",
        "ground_truth":    "According to SKYbrary's Engine Failure During Takeoff article, at V1 (the takeoff decision speed), the crew must decide to either Continue the Takeoff or initiate a Rejected Takeoff (RTO). V1 is the maximum speed at which an RTO can be initiated and the aircraft stopped within remaining runway. Above V1: the takeoff must be continued even with engine failure — apply full power on remaining engines, rotate at Vr, establish positive climb at V2, and follow single-engine climb procedures. Below V1: RTO should be initiated — close throttles, apply maximum braking, deploy speedbrakes, use reverse thrust if available.",
        "relevant_doc_ids": ["sky-1cdf236cdff2"],
        "difficulty":      "medium",
        "notes":           "V1 decision — continue vs. reject",
    },
    {
        "query_id":        "SKY-006",
        "source":          "SKYBRARY",
        "query_type":      "procedural",
        "question":        "According to SKYbrary, what are the standard components of fuel planning for a commercial flight?",
        "ground_truth":    "According to SKYbrary's Fuel Management article, the standard fuel planning components are: (1) Trip fuel — fuel required for the planned route; (2) Contingency fuel — typically 5% of trip fuel (EU OPS) or as calculated per operator policy; (3) Alternate fuel — fuel to fly from destination to the designated alternate airport; (4) Final reserve fuel — 30 minutes at holding speed for jets; (5) Additional fuel — as determined by the commander for specific conditions (holding, icing, MEL restrictions); (6) Extra fuel — at the commander's discretion. Minimum Fuel state means the aircraft must land at the nearest airport; Mayday Fuel means insufficient fuel to land safely.",
        "relevant_doc_ids": ["sky-7fb22bb63008"],
        "difficulty":      "medium",
        "notes":           "Fuel planning components",
    },
    {
        "query_id":        "SKY-007",
        "source":          "SKYBRARY",
        "query_type":      "cross_ref",
        "question":        "How does SKYbrary link the concept of Loss of Control In-Flight (LOC-I) to Upset Prevention and Recovery Training (UPRT)?",
        "ground_truth":    "SKYbrary's UPRT article explicitly states that LOC-I remains the leading cause of fatal accidents in commercial aviation, which is why UPRT was developed. LOC-I typically occurs when an aircraft enters an upset condition — pitch greater than 25° nose up, 10° nose down, or bank exceeding 45°. UPRT was mandated by regulators (including ICAO and FAA) specifically to address LOC-I by training pilots to recognize impending upsets and recover correctly. The SKYbrary LOC-I article identifies upset as one of its primary causes, while the UPRT article provides the training solution framework.",
        "relevant_doc_ids": ["sky-d5b855e045c3", "sky-96c52a0faa8e"],
        "difficulty":      "medium",
        "notes":           "LOC-I to UPRT regulatory linkage",
    },
    {
        "query_id":        "SKY-008",
        "source":          "SKYBRARY",
        "query_type":      "cross_ref",
        "question":        "According to SKYbrary, how do TCAS Resolution Advisories relate to ATC instructions, and what should a pilot do when they conflict?",
        "ground_truth":    "According to SKYbrary's ACAS/TCAS article, the most important TCAS operating rule is that pilots must ALWAYS follow a Resolution Advisory (RA) even if it conflicts with ATC instructions. When an RA is issued, the pilot must maneuver as directed by the RA and promptly inform ATC of the RA. ATC instructions must be disregarded until the RA has been resolved. This is because TCAS has real-time collision geometry data that ATC may not have. Once the RA is resolved, normal ATC communications resume. This rule supersedes standard pilot/ATC authority structure for the duration of the RA.",
        "relevant_doc_ids": ["sky-81fd0d261b1b"],
        "difficulty":      "hard",
        "notes":           "RA vs. ATC conflict resolution",
    },
    {
        "query_id":        "SKY-009",
        "source":          "SKYBRARY",
        "query_type":      "cross_ref",
        "question":        "How does SKYbrary's article on Threat and Error Management (TEM) relate to Situational Awareness?",
        "ground_truth":    "SKYbrary's TEM framework and Situational Awareness (SA) are directly linked as complementary safety concepts. TEM identifies three categories: Threats (external events increasing operational complexity), Errors (flight crew actions/inactions deviating from intentions), and Undesired States (resulting unsafe conditions). Maintaining SA — the three Endsley levels of perception, comprehension, and projection — is a primary defense against TEM threats becoming errors and errors becoming undesired states. Degraded SA (particularly Level 2 comprehension and Level 3 projection failure) is identified in both articles as a key pathway to accidents, including CFIT and LOC-I.",
        "relevant_doc_ids": ["sky-9b0bd0dea6ab", "sky-06767597f894"],
        "difficulty":      "hard",
        "notes":           "TEM-SA conceptual integration",
    },
    {
        "query_id":        "SKY-010",
        "source":          "SKYBRARY",
        "query_type":      "edge_case",
        "question":        "According to SKYbrary, what is the Time of Useful Consciousness at cruise altitude, and what does this mean for emergency descent timing?",
        "ground_truth":    "According to SKYbrary's Emergency Descent article, at cruise altitude (FL370-FL410), the Time of Useful Consciousness (TUC) — the time available for a crew member to take meaningful action before cognitive impairment — is extremely short: approximately 9-15 seconds at FL410 and 15-20 seconds at FL350. This means crew must don oxygen masks as an immediate memory action before TUC expires. The emergency descent must be initiated immediately after masking — any delay in executing the emergency descent checklist or waiting for ATC clearance risks incapacitation if oxygen fails.",
        "relevant_doc_ids": ["sky-fe966cb6b396"],
        "difficulty":      "hard",
        "notes":           "TUC — critical time constraint",
    },
    {
        "query_id":        "SKY-011",
        "source":          "SKYBRARY",
        "query_type":      "edge_case",
        "question":        "According to SKYbrary, what makes CFIT accidents particularly insidious compared to other accident types?",
        "ground_truth":    "According to SKYbrary's CFIT article, CFIT is particularly insidious because the aircraft is fully airworthy and under complete pilot control — there is no mechanical failure prompting crew action. The hazard is the crew's unawareness of proximity to terrain. CFIT most often occurs in the approach and landing phase, frequently on the centreline of an approach (meaning the crew believes they are correctly positioned). Many CFIT accidents occur because of loss of situational awareness in the vertical plane — crews believe they are higher than they actually are. The absence of any warning cue until impact leaves little or no time for recovery.",
        "relevant_doc_ids": ["sky-66ea9387bd89"],
        "difficulty":      "medium",
        "notes":           "CFIT — no warning paradigm",
    },
    {
        "query_id":        "SKY-012",
        "source":          "SKYBRARY",
        "query_type":      "edge_case",
        "question":        "According to SKYbrary, what are the special risks of Low Level Wind Shear during the departure phase compared to the approach phase?",
        "ground_truth":    "According to SKYbrary's Low Level Wind Shear article, while LLWS is hazardous in both phases, departure presents special risks: aircraft are at maximum weight (maximum takeoff weight), have minimum energy reserves, and are committed to the departure climb — there is no option to simply go around to an alternate. In departure, the primary risk is reduced terrain clearance and inability to achieve required obstacle clearance if airspeed suddenly drops. On approach, the primary risk is runway excursion if a headwind-to-tailwind shear causes excessive speed on touchdown; however, a go-around option exists. Departure shear is generally considered more dangerous because escape options are limited.",
        "relevant_doc_ids": ["sky-c98190cfb55b"],
        "difficulty":      "hard",
        "notes":           "Departure vs. approach LLWS risk asymmetry",
    },
    {
        "query_id":        "SKY-013",
        "source":          "SKYBRARY",
        "query_type":      "comparative",
        "question":        "According to SKYbrary, what is the difference between a Runway Excursion and a Runway Incursion?",
        "ground_truth":    "According to SKYbrary: A Runway Excursion is an event where an aircraft veers off or overruns the runway surface during takeoff or landing — the aircraft itself is the object that leaves the runway. Types include overrun (aircraft departs at the end of runway) and veer-off (aircraft departs the side). A Runway Incursion (per ICAO definition) is any occurrence at an aerodrome involving the incorrect presence of an aircraft, vehicle, or person on the protected area of a surface designated for landing and takeoff — it is a conflict on the runway involving multiple parties, where the aircraft may be either correctly or incorrectly positioned. Excursion: single aircraft problem. Incursion: collision/conflict risk between multiple parties on the runway.",
        "relevant_doc_ids": ["sky-cca6bb0969f0", "sky-434f55a08734"],
        "difficulty":      "easy",
        "notes":           "Definitional comparison",
    },
    {
        "query_id":        "SKY-014",
        "source":          "SKYBRARY",
        "query_type":      "comparative",
        "question":        "How does SKYbrary describe the difference between physical fatigue and mental fatigue in aviation, and which types of errors does each most affect?",
        "ground_truth":    "According to SKYbrary's Fatigue article: Physical fatigue is defined as the inability to exert force with muscles to the expected degree — affecting manual control tasks, physical checklists execution, and emergency physical procedures. Mental fatigue is defined as a general decrease of attention and ability to perform complex tasks with customary efficiency — affecting decision-making, situational awareness, communication clarity, and cognitive tasks. In aviation, mental fatigue is considered the greater hazard because most critical flight deck tasks are cognitive. Causes include loss/interruption of normal sleep patterns, shift patterns, and circadian rhythm disruption from time zone crossings. Cumulative fatigue (build-up over time from insufficient recovery) affects both types.",
        "relevant_doc_ids": ["sky-ba4c2b6d3dc4"],
        "difficulty":      "medium",
        "notes":           "Physical vs. mental fatigue distinction",
    },
    {
        "query_id":        "SKY-015",
        "source":          "SKYBRARY",
        "query_type":      "comparative",
        "question":        "According to SKYbrary, how does CRM (Crew Resource Management) differ from TEM (Threat and Error Management) as safety frameworks?",
        "ground_truth":    "According to SKYbrary: CRM is a management system focused on optimum use of all available resources — equipment, procedures, and people — to promote safety and enhance operational efficiency. It emphasizes skills: communication, situational awareness, problem solving, decision making, and teamwork. CRM is primarily a training and behavioral framework. TEM is a conceptual framework for understanding the interaction between safety and human performance — it provides a model for analyzing how threats, errors, and undesired aircraft states interact and where defenses can be placed. TEM is more of an analytical and operational awareness model. In practice, CRM skills are the tools that crews use to manage the threats and errors identified in the TEM model.",
        "relevant_doc_ids": ["sky-d943a2a0a798", "sky-9b0bd0dea6ab"],
        "difficulty":      "hard",
        "notes":           "CRM vs TEM framework comparison",
    },
]


# ── Utility functions ─────────────────────────────────────────────────────────


def get_queries_by_source(source: str) -> list[dict]:
    """Source ke saare queries return karo."""
    return [q for q in GOLDEN_QUERIES if q["source"] == source]


def get_queries_by_type(query_type: str) -> list[dict]:
    """Query type ke saare queries return karo."""
    return [q for q in GOLDEN_QUERIES if q["query_type"] == query_type]


def get_queries_by_difficulty(difficulty: str) -> list[dict]:
    """Difficulty level ke saare queries return karo."""
    return [q for q in GOLDEN_QUERIES if q["difficulty"] == difficulty]


def print_stats() -> None:
    """Query distribution print karo."""
    from collections import Counter

    print("=" * 60)
    print(f"{'SkyLex Golden Queries — Stats':^60}")
    print("=" * 60)

    print(f"\nTotal queries: {len(GOLDEN_QUERIES)}")

    by_source = Counter(q["source"] for q in GOLDEN_QUERIES)
    print("\nBy Source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src:<12} : {count}")

    by_type = Counter(q["query_type"] for q in GOLDEN_QUERIES)
    print("\nBy Query Type:")
    for qt, count in sorted(by_type.items()):
        print(f"  {qt:<15} : {count}")

    by_diff = Counter(q["difficulty"] for q in GOLDEN_QUERIES)
    print("\nBy Difficulty:")
    for d, count in sorted(by_diff.items()):
        print(f"  {d:<8} : {count}")

    print("=" * 60)


if __name__ == "__main__":
    print_stats()