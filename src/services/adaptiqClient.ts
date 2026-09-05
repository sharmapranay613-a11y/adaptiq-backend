/**
 * @license
 * AdaptiQ Client SDK
 *
 * Reusable AI layer client for Learnova AI.
 * Connects to the deployed AdaptiQ backend.
 */

import type {
  GenerateQuestionsParams,
  GenerateQuestionsResult,
  AnalyzeMistakeParams,
  AnalyzeMistakeResult,
  ExplainTopicParams,
  ExplainTopicResult,
  TutorParams,
  TutorResult,
  ApiResponse,
  TestResultItem,
} from '../types/adaptiq';

// Deployed AdaptiQ backend URL
const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  console.warn(
    'NEXT_PUBLIC_API_URL is not configured. Please add it to .env.local.'
  );
}

/**
 * Generic POST request helper
 */
async function postJson<T>(
  endpoint: string,
  payload: unknown
): Promise<T> {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const json: ApiResponse<T> = await response.json();

  if (!response.ok || !json.success || !json.data) {
    throw new Error(
      json.error || `Request failed with status ${response.status}`
    );
  }

  return json.data;
}

/**
 * 1. Generate AI Questions
 */
export async function generateQuestions(
  params: GenerateQuestionsParams
): Promise<GenerateQuestionsResult> {
  return postJson<GenerateQuestionsResult>(
    `${API_URL}/api/ai/generate-questions`,
    params
  );
}

/**
 * 2. Analyze Student Mistake
 */
export async function analyzeMistake(
  params: AnalyzeMistakeParams
): Promise<AnalyzeMistakeResult> {
  return postJson<AnalyzeMistakeResult>(
    `${API_URL}/api/ai/analyze-mistake`,
    params
  );
}

/**
 * 3. Explain Topic
 */
export async function explainTopic(
  params: ExplainTopicParams
): Promise<ExplainTopicResult> {
  return postJson<ExplainTopicResult>(
    `${API_URL}/api/ai/explain-topic`,
    params
  );
}

/**
 * 4. AI Tutor
 */
export async function tutor(
  params: TutorParams
): Promise<TutorResult> {
  return postJson<TutorResult>(
    `${API_URL}/api/ai/tutor`,
    params
  );
}

/**
 * 5. Run AI backend tests
 */
export async function runJavaInheritanceTests(): Promise<{
  success: boolean;
  allPassed: boolean;
  results: TestResultItem[];
}> {
  const response = await fetch(
    `${API_URL}/api/ai/run-tests`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  return response.json();
}

/**
 * 6. Check AI backend health
 */
export async function checkAiHealth(): Promise<{
  status: string;
  service: string;
  geminiConfigured: boolean;
  timestamp: string;
}> {
  const response = await fetch(
    `${API_URL}/api/ai/health`
  );

  return response.json();
}