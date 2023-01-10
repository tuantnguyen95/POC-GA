from prometheus_client import Counter, Summary

AI_SERVICE_ELEMENT_COMPARISON_TOTAL = Counter(
  "ai_service_element_comparison_total", "Total number of element comparison requests"
)
AI_SERVICE_ELEMENT_COMPARISON_SUCCESSFUL_TOTAL = Counter(
  "ai_service_element_comparison_successful_total",
  "Total number of successful element comparison requests",
)
AI_SERVICE_ELEMENT_COMPARISON_LATENCY_SECONDS = Summary(
  "ai_service_element_comparison_latency_seconds",
  "Average time spent on an element comparison request in second ",
)
AI_SERVICE_ELEMENT_FINDING_TOTAL = Counter(
  "ai_service_element_finding_total", "Total number of element finding requests"
)
AI_SERVICE_ELEMENT_FINDING_SUCCESSFUL_TOTAL = Counter(
  "ai_service_element_finding_successful_total",
  "Total number of successful element finding requests",
)
AI_SERVICE_ELEMENT_FINDING_LATENCY_SECONDS = Summary(
  "ai_service_element_finding_latency_seconds",
  "Average time spent on an element finding request in second ",
)
