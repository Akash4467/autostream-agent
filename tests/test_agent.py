"""
tests/test_agent.py

Unit tests for RAG, lead capture tool, and intent classifier.
Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short --cov=agent --cov=tools
"""

from __future__ import annotations

import pytest
from agent.rag import load_knowledge_base, build_kb_context
from tools.lead_capture import mock_lead_capture, get_all_leads
from agent.graph import classify_intent


# ── RAG ───────────────────────────────────────────────────────────────────────

class TestRAG:
    def test_kb_loads_successfully(self):
        kb = load_knowledge_base()
        assert isinstance(kb, dict)

    def test_kb_has_both_plans(self):
        kb = load_knowledge_base()
        assert "basic" in kb["plans"] and "pro" in kb["plans"]

    def test_basic_plan_price(self):
        assert load_knowledge_base()["plans"]["basic"]["price_monthly"] == 29

    def test_pro_plan_price(self):
        assert load_knowledge_base()["plans"]["pro"]["price_monthly"] == 79

    def test_pro_has_ai_captions(self):
        features = " ".join(load_knowledge_base()["plans"]["pro"]["features"]).lower()
        assert "caption" in features

    def test_pro_has_4k(self):
        features = " ".join(load_knowledge_base()["plans"]["pro"]["features"]).lower()
        assert "4k" in features

    def test_pro_has_unlimited_videos(self):
        features = " ".join(load_knowledge_base()["plans"]["pro"]["features"]).lower()
        assert "unlimited" in features

    def test_refund_policy_mentions_7_days(self):
        assert "7 days" in load_knowledge_base()["policies"]["refund_policy"]

    def test_context_contains_prices(self):
        ctx = build_kb_context()
        assert "$29" in ctx and "$79" in ctx

    def test_context_contains_policies(self):
        assert "refund" in build_kb_context().lower()

    def test_context_contains_supported_platforms(self):
        ctx = build_kb_context()
        assert "YouTube" in ctx and "TikTok" in ctx


# ── Lead capture ──────────────────────────────────────────────────────────────

class TestLeadCapture:
    def test_returns_success_true(self):
        r = mock_lead_capture("Test User", "test@example.com", "YouTube")
        assert r["success"] is True

    def test_message_contains_name(self):
        r = mock_lead_capture("Alice Smith", "alice@example.com", "YouTube")
        assert "Alice Smith" in r["message"]

    def test_lead_id_format(self):
        r = mock_lead_capture("Bob Jones", "bob@test.com", "Instagram")
        assert r["lead_id"].startswith("LEAD-")

    def test_lead_stored_in_memory(self):
        before = len(get_all_leads())
        mock_lead_capture("Carol White", "carol@test.com", "TikTok")
        assert len(get_all_leads()) == before + 1

    def test_lead_has_timestamp(self):
        r = mock_lead_capture("Dave Brown", "dave@test.com", "LinkedIn")
        assert "captured_at" in r["lead"]
        assert r["lead"]["captured_at"].endswith("Z")

    def test_lead_has_status_new(self):
        r = mock_lead_capture("Eve Green", "eve@test.com", "Twitch")
        assert r["lead"]["status"] == "new"

    def test_lead_fields_persisted(self):
        r = mock_lead_capture("Frank Hill", "frank@test.com", "Facebook")
        assert r["lead"]["name"] == "Frank Hill"
        assert r["lead"]["email"] == "frank@test.com"
        assert r["lead"]["platform"] == "Facebook"

    def test_prints_confirmation(self, capsys):
        mock_lead_capture("Grace Lee", "grace@test.com", "YouTube")
        out = capsys.readouterr().out
        assert "Grace Lee" in out
        assert "grace@test.com" in out
        assert "YouTube" in out


# ── Intent classifier ─────────────────────────────────────────────────────────

class TestIntentClassifier:
    # ── Greetings ────────────────────────────────────────────────────────────
    def test_hi(self):      assert classify_intent("Hi there!")    == "greeting"
    def test_hello(self):   assert classify_intent("Hello!")        == "greeting"
    # Pure greeting — no other keywords present
    def test_hey(self):     assert classify_intent("Hey there!")    == "greeting"

    # ── Pricing inquiry ──────────────────────────────────────────────────────
    # "pricing" and "plans" are pricing keywords → pricing_inquiry
    def test_pricing_question(self):
        assert classify_intent("What are your pricing plans?")         == "pricing_inquiry"
    # "plan" is a pricing keyword → pricing_inquiry
    def test_feature_question(self):
        assert classify_intent("Tell me about the Pro plan features")  == "pricing_inquiry"

    # ── Support ──────────────────────────────────────────────────────────────
    # "refund" is a support keyword → support
    def test_refund_question(self):
        assert classify_intent("What is your refund policy?")          == "support"
    # "help" is a support keyword even when combined with a greeting
    def test_help_request(self):
        assert classify_intent("Hey, can you help me?")                == "support"
    # "cancel" → support
    def test_cancel(self):
        assert classify_intent("I want to cancel my subscription")     == "support"

    # ── High intent ──────────────────────────────────────────────────────────
    def test_sign_up(self):
        assert classify_intent("I want to sign up for the Pro plan")   == "high_intent"
    def test_buy(self):
        assert classify_intent("I'd like to buy the Pro plan")         == "high_intent"
    def test_get_started(self):
        assert classify_intent("I'm ready to get started")             == "high_intent"
    def test_sounds_good(self):
        assert classify_intent("sounds good, let's go")                == "high_intent"
    def test_subscribe(self):
        assert classify_intent("I want to subscribe")                  == "high_intent"
    def test_upgrade(self):
        assert classify_intent("I want to upgrade to Pro")             == "high_intent"

    # ── Lead collection ──────────────────────────────────────────────────────
    def test_email_provided(self):
        assert classify_intent("my email is alex@example.com")         == "lead_collection"
    def test_name_provided(self):
        assert classify_intent("my name is Alex")                      == "lead_collection"
    def test_platform_provided(self):
        assert classify_intent("I mainly post on YouTube")             == "lead_collection"

    # ── General product inquiry (fallback) ───────────────────────────────────
    def test_generic_question(self):
        assert classify_intent("What does AutoStream do?")             == "product_inquiry"
    def test_feature_no_plan_keyword(self):
        assert classify_intent("What features do you have?")           == "product_inquiry"
