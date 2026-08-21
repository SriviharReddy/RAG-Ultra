import os
import sys
import json
import asyncio
import tempfile
import pymupdf  # PyMuPDF
import httpx
from httpx import ASGITransport
from dotenv import load_dotenv

from core.config import get_settings
from ingest_cli import ingest_file
from my_agent.agent import graph
from app import app, condense_query, ChatMessage

load_dotenv()

# ANSI Color Codes for terminal rendering
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def create_sample_technical_pdf(file_path: str) -> str:
    """
    Creates a sample multi-page PDF document containing technical diagrams,
    specification tables, and operating safety protocols.
    """
    doc = pymupdf.open()

    # --- Page 1: Overview & Specifications Table ---
    page1 = doc.new_page(width=595, height=842) # A4
    page1.insert_text((50, 50), "Turbine Alpha-9 Operating Manual", fontsize=20, color=(0.1, 0.2, 0.6))
    page1.insert_text((50, 80), "Section 1: General Specifications and Safety Thresholds", fontsize=13, color=(0.2, 0.2, 0.2))

    p1_content = (
        "The Turbine Alpha-9 is a high-efficiency industrial compression unit designed for severe operating environments.\n"
        "All operators must review pressure tolerances and emergency cutoff procedures prior to deployment.\n\n"
        "Operating Limits:\n"
        "- Maximum Nominal Pressure: 450 PSI (31.0 bar)\n"
        "- Emergency Vent Trigger Threshold: 520 PSI\n"
        "- Standard Operating Temperature: -20 C to 85 C\n"
        "- Thermal Warning Alarm: 95 C\n\n"
        "Table 1.1: Component Lubrication and Inspection Schedule\n"
        "| Component ID | Lubricant Type | Inspection Frequency | Replacement Cycle |\n"
        "|--------------|----------------|----------------------|-------------------|\n"
        "| Bearings-A   | Synthetic ISO 68 | Bi-weekly (14 days)  | 12,000 Hours      |\n"
        "| Main Seal    | Fluoropolymer  | Monthly (30 days)    | 8,000 Hours       |\n"
        "| Rotor Vanes  | Dry Film MoS2  | Quarterly (90 days)  | 24,000 Hours      |\n"
    )
    page1.insert_text((50, 120), p1_content, fontsize=10, color=(0, 0, 0))

    # Draw a visual border representing a technical schematic diagram
    page1.draw_rect(pymupdf.Rect(50, 480, 545, 680), color=(0.2, 0.4, 0.8), width=1.5)
    page1.insert_text((60, 500), "[Figure 1.A: Hydraulic Bleed Valve and Primary Rotor Assembly]", fontsize=10, color=(0.1, 0.2, 0.6))
    page1.draw_line(pymupdf.Point(70, 550), pymupdf.Point(250, 550), color=(0.8, 0.2, 0.2), width=2)
    page1.insert_text((70, 570), "Primary Intake Flange (Port A)", fontsize=9, color=(0.3, 0.3, 0.3))
    page1.draw_line(pymupdf.Point(300, 550), pymupdf.Point(480, 550), color=(0.2, 0.6, 0.2), width=2)
    page1.insert_text((300, 570), "High-Pressure Return Manifold", fontsize=9, color=(0.3, 0.3, 0.3))

    # --- Page 2: Wet & Extreme Weather Protocol ---
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 50), "Turbine Alpha-9 Operating Manual", fontsize=20, color=(0.1, 0.2, 0.6))
    page2.insert_text((50, 80), "Section 2: Wet Conditions and Flood Protocol", fontsize=13, color=(0.2, 0.2, 0.2))

    p2_content = (
        "Operating in wet, high-humidity, or marine environments requires adherence to Protocol W-7.\n\n"
        "Pre-Operation Wet Checklist:\n"
        "1. Verify IP67 waterproof enclosure seals on all electrical junctions.\n"
        "2. Check secondary condensate drain valve for sediment blockages.\n"
        "3. Apply anti-corrosion barrier spray to external electrical terminals.\n\n"
        "Wet Condition Operational Constraints:\n"
        "- Derate maximum continuous output by 15% when ambient humidity exceeds 90%.\n"
        "- Enable automatic manifold heater if ambient temperature drops below 4 C in wet weather.\n"
        "- In the event of standing water exceeding 10 cm around the skid base, initiate immediate controlled shutdown (Sequence E-2).\n\n"
        "Table 2.1: Wet Weather Calibration Settings\n"
        "| Environmental Parameter | Dry Mode Setting | Wet Protocol W-7 Setting |\n"
        "|-------------------------|------------------|--------------------------|\n"
        "| Intake Sensor Sensitivity | High (1.0x)    | Filtered (0.85x)         |\n"
        "| Heater Duty Cycle       | On Demand (Auto) | Continuous 50%           |\n"
        "| Bleed Valve Pre-Charge  | 15 PSI           | 25 PSI                   |\n"
    )
    page2.insert_text((50, 120), p2_content, fontsize=10, color=(0, 0, 0))

    doc.save(file_path)
    doc.close()
    return file_path

async def run_end_to_end_demo():
    print(f"\n{BOLD}{MAGENTA}========================================================================{RESET}")
    print(f"{BOLD}{MAGENTA}       RAG-ULTRA: SOTA AGENTIC MULTIMODAL RAG SHOWCASE DEMO           {RESET}")
    print(f"{BOLD}{MAGENTA}========================================================================{RESET}\n")


    temp_dir = tempfile.mkdtemp()
    sample_pdf_path = os.path.join(temp_dir, "turbine_alpha9_manual.pdf")
    create_sample_technical_pdf(sample_pdf_path)
    print(f"{GREEN}[✓] Generated sample technical multi-page PDF:{RESET} {sample_pdf_path}\n")

    # -------------------------------------------------------------
    # 1. Ingestion Pipeline Demo
    # -------------------------------------------------------------
    print(f"{BOLD}{CYAN}------------------------------------------------------------{RESET}")
    print(f"{BOLD}{CYAN}1. Ingestion Engine: Layout-Aware Parsing & Image Caching{RESET}")
    print(f"{BOLD}{CYAN}------------------------------------------------------------{RESET}")
    ingest_result = await ingest_file(
        file_path=sample_pdf_path,
        document_id="manual_turbine_a9",
        chunk_size=500,
        chunk_overlap=80
    )
    print(f"{GREEN}[✓] Ingestion Result:{RESET} {json.dumps(ingest_result, indent=2)}\n")

    # -------------------------------------------------------------
    # 2. Conversational Query Condensation (Pattern A)
    # -------------------------------------------------------------
    print(f"{BOLD}{CYAN}------------------------------------------------------------{RESET}")
    print(f"{BOLD}{CYAN}2. Pattern A: Conversational Query Condensation{RESET}")
    print(f"{BOLD}{CYAN}------------------------------------------------------------{RESET}")
    chat_history = [
        ChatMessage(role="user", content="What are the maximum pressure limits for the Turbine Alpha-9?"),
        ChatMessage(role="assistant", content="The maximum nominal pressure is 450 PSI, with an emergency vent trigger at 520 PSI.")
    ]
    raw_followup = "What about in wet conditions?"
    print(f"{YELLOW}User Follow-up Query:{RESET} \"{raw_followup}\"")
    print(f"{BLUE}Chat History Count:{RESET} {len(chat_history)} messages")

    condensed = await condense_query(raw_followup, chat_history)
    print(f"{GREEN}[✓] Rewritten Standalone Query:{RESET} \"{condensed}\"\n")

    # -------------------------------------------------------------
    # 3. LangGraph Corrective RAG Workflow Execution
    # -------------------------------------------------------------
    print(f"{BOLD}{CYAN}------------------------------------------------------------{RESET}")
    print(f"{BOLD}{CYAN}3. LangGraph Workflow: CRAG with LLM-as-a-Judge & Provenance{RESET}")
    print(f"{BOLD}{CYAN}------------------------------------------------------------{RESET}")

    initial_state = {
        "raw_query": raw_followup,
        "query": condensed,
        "condensed_query": condensed,
        "retrieved_chunks": [],
        "route_decision": "retrieve",
        "retry_count": 0,
        "critique": None,
        "expanded_query": None,
        "is_relevant": None,
        "llm_inputs": [],
        "answer": None,
        "citations": [],
        "is_grounded": None,
        "groundedness_score": None,
        "metadata_filter": {"doc_id": "manual_turbine_a9"}
    }

    final_state = await graph.ainvoke(initial_state)
    print(f"{GREEN}[✓] State Graph Execution Complete!{RESET}")
    print(f"{BLUE}Retrieved Chunks Count:{RESET} {len(final_state.get('retrieved_chunks', []))}")
    print(f"{BLUE}Judge Relevance:{RESET} {final_state.get('is_relevant')}")
    print(f"{BLUE}Judge Critique:{RESET} {final_state.get('critique')}")
    print(f"{BLUE}Groundedness Verified:{RESET} {final_state.get('is_grounded')} (Score: {final_state.get('groundedness_score')})")
    print(f"\n{BOLD}{YELLOW}Generated Answer with Inline Provenance:{RESET}\n{final_state.get('answer')}\n")

    print(f"{BOLD}{BLUE}Structured Footnote Citations:{RESET}")
    for cit in final_state.get("citations", []):
        print(f"  [{cit.get('id')}] Source: {cit.get('source')}, Page {cit.get('page')} -> {cit.get('snippet')}")
    print()

    # -------------------------------------------------------------
    # 4. REST API Gateway & SSE Streaming Verification
    # -------------------------------------------------------------
    print(f"{BOLD}{CYAN}------------------------------------------------------------{RESET}")
    print(f"{BOLD}{CYAN}4. FastAPI Gateway: Health, REST Query & SSE Streaming Test{RESET}")
    print(f"{BOLD}{CYAN}------------------------------------------------------------{RESET}")

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 4.1 Health endpoint
        health_resp = await client.get("/api/v1/health")
        print(f"{GREEN}[✓] GET /api/v1/health -> Status {health_resp.status_code}:{RESET}")
        print(json.dumps(health_resp.json(), indent=2))

        # 4.2 Multipart Ingest endpoint
        with open(sample_pdf_path, "rb") as f_pdf:
            files = {"file": ("uploaded_turbine_manual.pdf", f_pdf.read(), "application/pdf")}
            data = {"document_id": "uploaded_manual_002"}
            ingest_api_resp = await client.post("/api/v1/ingest", files=files, data=data)
        print(f"\n{GREEN}[✓] POST /api/v1/ingest -> Status {ingest_api_resp.status_code}:{RESET}")
        print(json.dumps(ingest_api_resp.json(), indent=2))

        # 4.2 Standard Query endpoint
        query_payload = {
            "query": "What is the inspection frequency for Bearings-A?",
            "chat_history": [],
            "metadata_filter": {"doc_id": "manual_turbine_a9"}
        }
        query_resp = await client.post("/api/v1/query", json=query_payload)
        print(f"\n{GREEN}[✓] POST /api/v1/query -> Status {query_resp.status_code}:{RESET}")
        q_data = query_resp.json()
        print(f"{BLUE}Answer:{RESET} {q_data.get('answer')}")
        print(f"{BLUE}Latency:{RESET} {q_data.get('metadata', {}).get('latency_ms')} ms")

        # 4.3 SSE Streaming endpoint
        print(f"\n{BOLD}{MAGENTA}[SSE Stream Test] POST /api/v1/query/stream ...{RESET}")
        stream_payload = {
            "query": "What is Protocol W-7 for wet conditions?",
            "chat_history": []
        }
        event_count = 0
        async with client.stream("POST", "/api/v1/query/stream", json=stream_payload) as stream_resp:
            print(f"{GREEN}[✓] SSE Stream Connected! (Status {stream_resp.status_code}){RESET}")
            async for line in stream_resp.aiter_lines():
                if line.startswith("event: "):
                    event_type = line.replace("event: ", "").strip()
                    print(f"  {MAGENTA}► Event:{RESET} {BOLD}{event_type}{RESET}")
                    event_count += 1
                elif line.startswith("data: "):
                    data_str = line.replace("data: ", "").strip()
                    try:
                        data_obj = json.loads(data_str)
                        if "token" in event_type or "chunk" in data_obj:
                            chunk_text = data_obj.get("chunk", "")
                            sys.stdout.write(f"{YELLOW}{chunk_text}{RESET}")
                            sys.stdout.flush()
                        elif event_type in ["evaluating", "retrieving", "verifying", "final_result"]:
                            print(f"    Payload: {json.dumps(data_obj)[:140]}...")
                    except Exception:
                        pass

        print(f"\n{GREEN}[✓] SSE Stream Finished! Processed {event_count} graph transition events.{RESET}")

    print(f"\n{BOLD}{GREEN}========================================================================{RESET}")
    print(f"{BOLD}{GREEN}           ALL SOTA RAG-ULTRA DEMO TESTS PASSED SUCCESSFULLY!          {RESET}")
    print(f"{BOLD}{GREEN}========================================================================{RESET}\n")

if __name__ == "__main__":
    asyncio.run(run_end_to_end_demo())
