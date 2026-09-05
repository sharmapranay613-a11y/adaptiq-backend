/**
 * AdaptiQ AI Service Layer
 * Powered by Google Gemini API (@google/genai)
 */

import { GoogleGenAI, Type } from '@google/genai'

import type {
  GenerateQuestionsParams,
  GenerateQuestionsResult,
  AnalyzeMistakeParams,
  AnalyzeMistakeResult,
  ExplainTopicParams,
  ExplainTopicResult,
  TutorParams,
  TutorResult,
  MistakeType,
} from '../src/types/adaptiq'

// ============================================================
// GEMINI CLIENT
// ============================================================

let aiClient: GoogleGenAI | null = null

function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY

    if (!apiKey) {
      throw new Error(
        'GEMINI_API_KEY is not configured in the server environment variables. Please check the Secrets panel.'
      )
    }

    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  }

  return aiClient
}

// Keep the current working model
const DEFAULT_MODEL = 'gemini-3.1-flash-lite' 

// ============================================================
// RETRY HELPER
// ============================================================

async function callWithRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3
): Promise<T> {
  let attempt = 0

  while (attempt < maxRetries) {
    try {
      return await fn()
    } catch (err: any) {
      attempt++

      const errorMessage = String(
        err?.message || err || ''
      )

      // Do NOT retry an exhausted quota.
      const isQuotaExhausted =
        errorMessage.includes('RESOURCE_EXHAUSTED') ||
        errorMessage.includes('quota')

      if (isQuotaExhausted) {
        throw err
      }

      const isTransient =
        errorMessage.includes('503') ||
        errorMessage.includes('429') ||
        errorMessage.includes('UNAVAILABLE') ||
        errorMessage.includes('high demand') ||
        errorMessage.includes('fetch failed')

      if (isTransient && attempt < maxRetries) {
        const delay =
          Math.pow(2, attempt) * 1000 +
          Math.random() * 500

        await new Promise((resolve) =>
          setTimeout(resolve, delay)
        )

        continue
      }

      throw err
    }
  }

  throw new Error('Max retries reached')
}

// ============================================================
// BATCH QUESTION TYPES
// ============================================================

export interface BatchQuestionRequest {
  subject: string
  topic: string
  difficulty: string
  count: number
}

export interface BatchQuestionItem {
  subject: string
  topic: string
  difficulty: string
  question: string
  options: string[]
  correctAnswer: string
  explanation: string
}

export interface BatchQuestionResult {
  questions: BatchQuestionItem[]
}

// ============================================================
// 1. NORMAL QUESTION GENERATION
// ============================================================

export async function generateQuestions(
  params: GenerateQuestionsParams
): Promise<GenerateQuestionsResult> {
  const {
    subject,
    topic,
    difficulty,
    count = 3,
  } = params

  const ai = getGeminiClient()

  const prompt = `Generate ${count} high-quality multiple choice question(s) for the subject "${subject}", on the topic "${topic}" at "${difficulty}" difficulty level.

Each question must:
- Have exactly 4 clear options.
- Specify the exact string of the correct answer.
- The correct answer must be one of the 4 options.
- Provide a concise educational explanation.
- Avoid duplicate questions.`

  const response = await callWithRetry(() =>
    ai.models.generateContent({
      model: DEFAULT_MODEL,
      contents: prompt,
      config: {
        systemInstruction:
          'You are AdaptiQ, an expert educational assessment AI. Always generate accurate, pedagogically sound, structured questions in valid JSON.',

        responseMimeType: 'application/json',

        responseSchema: {
          type: Type.OBJECT,

          properties: {
            questions: {
              type: Type.ARRAY,

              description:
                'Array of generated questions',

              items: {
                type: Type.OBJECT,

                properties: {
                  question: {
                    type: Type.STRING,
                    description:
                      'The question text',
                  },

                  options: {
                    type: Type.ARRAY,

                    items: {
                      type: Type.STRING,
                    },

                    description:
                      'Four multiple-choice options',
                  },

                  correctAnswer: {
                    type: Type.STRING,

                    description:
                      'The exact text of the correct option',
                  },

                  explanation: {
                    type: Type.STRING,

                    description:
                      'Clear pedagogical explanation of why this answer is correct',
                  },
                },

                required: [
                  'question',
                  'options',
                  'correctAnswer',
                  'explanation',
                ],
              },
            },
          },

          required: ['questions'],
        },
      },
    })
  )

  const text =
    response.text?.trim() || '{}'

  const parsed =
    JSON.parse(text) as GenerateQuestionsResult

  if (
    !parsed.questions ||
    !Array.isArray(parsed.questions)
  ) {
    throw new Error(
      'Invalid response structure received from Gemini API'
    )
  }

  return parsed
}

// ============================================================
// 2. BATCH QUESTION GENERATION
// ============================================================

/**
 * Generates diagnostic questions for multiple subjects
 * using ONE Gemini API request.
 *
 * This is the function used by the new diagnostic flow.
 */
export async function generateBatchQuestions(
  requests: BatchQuestionRequest[]
): Promise<BatchQuestionResult> {
  if (!requests.length) {
    throw new Error(
      'At least one subject/topic is required.'
    )
  }

  const ai = getGeminiClient()

  const totalQuestions =
    requests.reduce(
      (sum, item) =>
        sum + Math.max(1, item.count),
      0
    )

  const subjectInstructions =
    requests
      .map(
        (item, index) =>
          `${index + 1}. Subject: "${item.subject}"
Topic: "${item.topic}"
Difficulty: "${item.difficulty}"
Number of questions: ${item.count}`
      )
      .join('\n\n')

  const prompt = `Generate exactly ${totalQuestions} high-quality multiple-choice diagnostic questions for a student.

Use the following subject/topic requirements:

${subjectInstructions}

IMPORTANT RULES:

1. Generate EXACTLY the requested number of questions.
2. Follow the requested number for each subject.
3. Every question must have exactly 4 options.
4. correctAnswer must exactly match one of the options.
5. Every question must include its subject.
6. Every question must include its topic.
7. Every question must include its difficulty.
8. Do not generate duplicate questions.
9. Questions should test actual understanding.
10. Keep explanations concise and educational.
11. Return only valid JSON matching the schema.`

  const response = await callWithRetry(() =>
    ai.models.generateContent({
      model: DEFAULT_MODEL,

      contents: prompt,

      config: {
        systemInstruction:
          'You are AdaptiQ, an expert educational assessment AI. Generate accurate, balanced and pedagogically useful diagnostic questions. Follow the requested JSON schema exactly.',

        responseMimeType: 'application/json',

        responseSchema: {
          type: Type.OBJECT,

          properties: {
            questions: {
              type: Type.ARRAY,

              description:
                'All generated diagnostic questions',

              items: {
                type: Type.OBJECT,

                properties: {
                  subject: {
                    type: Type.STRING,
                    description:
                      'Subject of the question',
                  },

                  topic: {
                    type: Type.STRING,
                    description:
                      'Topic of the question',
                  },

                  difficulty: {
                    type: Type.STRING,
                    description:
                      'Difficulty level',
                  },

                  question: {
                    type: Type.STRING,
                    description:
                      'Question text',
                  },

                  options: {
                    type: Type.ARRAY,

                    items: {
                      type: Type.STRING,
                    },

                    description:
                      'Exactly four answer options',
                  },

                  correctAnswer: {
                    type: Type.STRING,

                    description:
                      'Exact text of the correct option',
                  },

                  explanation: {
                    type: Type.STRING,

                    description:
                      'Concise explanation of the correct answer',
                  },
                },

                required: [
                  'subject',
                  'topic',
                  'difficulty',
                  'question',
                  'options',
                  'correctAnswer',
                  'explanation',
                ],
              },
            },
          },

          required: ['questions'],
        },
      },
    })
  )

  const text =
    response.text?.trim() || '{}'

  const parsed =
    JSON.parse(text) as BatchQuestionResult

  if (
    !parsed.questions ||
    !Array.isArray(parsed.questions)
  ) {
    throw new Error(
      'Invalid batch question response from Gemini API'
    )
  }

  const validQuestions =
    parsed.questions.filter(
      (question) =>
        question.question &&
        question.subject &&
        question.topic &&
        Array.isArray(question.options) &&
        question.options.length === 4 &&
        question.correctAnswer &&
        question.explanation
    )

  if (!validQuestions.length) {
    throw new Error(
      'Gemini returned no valid diagnostic questions'
    )
  }

  return {
    questions: validQuestions,
  }
}

// ============================================================
// 3. ANALYZE MISTAKE
// ============================================================

export async function analyzeMistake(
  params: AnalyzeMistakeParams
): Promise<AnalyzeMistakeResult> {
  const {
    question,
    studentAnswer,
    correctAnswer,
    topic,
  } = params

  const ai = getGeminiClient()

  const validMistakeTypes: MistakeType[] = [
    'Conceptual',
    'Calculation',
    'Careless',
    'Question misunderstanding',
    'Memory/recall',
  ]

  const prompt = `Analyze this student error on the topic "${topic}":

Question: "${question}"

Student Answer: "${studentAnswer}"

Correct Answer: "${correctAnswer}"

Classify the mistake type strictly into one of the following 5 categories:

- Conceptual
- Calculation
- Careless
- Question misunderstanding
- Memory/recall

Provide a constructive explanation of what went wrong and an actionable recommendation for the student.`

  const response = await callWithRetry(() =>
    ai.models.generateContent({
      model: DEFAULT_MODEL,

      contents: prompt,

      config: {
        systemInstruction:
          'You are AdaptiQ Learning Diagnostic Engine. Accurately diagnose student misconceptions with precision and empathy.',

        responseMimeType: 'application/json',

        responseSchema: {
          type: Type.OBJECT,

          properties: {
            mistakeType: {
              type: Type.STRING,

              enum: validMistakeTypes,

              description:
                'Classification of the error',
            },

            explanation: {
              type: Type.STRING,

              description:
                'Concise explanation of the student misconception',
            },

            recommendation: {
              type: Type.STRING,

              description:
                'Actionable study recommendation',
            },
          },

          required: [
            'mistakeType',
            'explanation',
            'recommendation',
          ],
        },
      },
    })
  )

  const text =
    response.text?.trim() || '{}'

  const parsed =
    JSON.parse(text) as AnalyzeMistakeResult

  if (
    !validMistakeTypes.includes(
      parsed.mistakeType
    )
  ) {
    parsed.mistakeType = 'Conceptual'
  }

  return parsed
}

// ============================================================
// 4. EXPLAIN TOPIC
// ============================================================

export async function explainTopic(
  params: ExplainTopicParams
): Promise<ExplainTopicResult> {
  const {
    topic,
    studentLevel = 'intermediate',
  } = params

  const ai = getGeminiClient()

  const prompt = `Explain the topic "${topic}" tailored specifically for a "${studentLevel}" level student.

Provide:

1. simpleExplanation
2. example
3. keyPoints with 2 to 4 points
4. practiceQuestion`

  const response = await callWithRetry(() =>
    ai.models.generateContent({
      model: DEFAULT_MODEL,

      contents: prompt,

      config: {
        systemInstruction:
          'You are AdaptiQ Explainer AI. Break down complex topics into intuitive, easy-to-grasp concepts with practical examples.',

        responseMimeType: 'application/json',

        responseSchema: {
          type: Type.OBJECT,

          properties: {
            simpleExplanation: {
              type: Type.STRING,
            },

            example: {
              type: Type.STRING,
            },

            keyPoints: {
              type: Type.ARRAY,

              items: {
                type: Type.STRING,
              },
            },

            practiceQuestion: {
              type: Type.STRING,
            },
          },

          required: [
            'simpleExplanation',
            'example',
            'keyPoints',
            'practiceQuestion',
          ],
        },
      },
    })
  )

  const text =
    response.text?.trim() || '{}'

  const parsed =
    JSON.parse(text) as ExplainTopicResult

  if (
    !parsed.simpleExplanation ||
    !Array.isArray(parsed.keyPoints)
  ) {
    throw new Error(
      'Invalid explanation structure from Gemini API'
    )
  }

  return parsed
}

// ============================================================
// 5. AI TUTOR
// ============================================================

export async function tutor(
  params: TutorParams
): Promise<TutorResult> {
  const {
    question,
    context = '',
  } = params

  const ai = getGeminiClient()

  const prompt = `Student Question: "${question}"

${
  context
    ? `Additional Context/Prior Learning: "${context}"`
    : ''
}

Provide a structured, scaffolded Socratic tutoring response following these 4 stages:

1. Hint
2. Simple Explanation
3. Example
4. Practice Question

Keep each section concise and punchy.`

  const response = await callWithRetry(() =>
    ai.models.generateContent({
      model: DEFAULT_MODEL,

      contents: prompt,

      config: {
        systemInstruction:
          'You are AdaptiQ Socratic Tutor. Scaffold learning progressively. Never overwhelm the student with long walls of text.',

        responseMimeType: 'application/json',

        responseSchema: {
          type: Type.OBJECT,

          properties: {
            hint: {
              type: Type.STRING,
            },

            simpleExplanation: {
              type: Type.STRING,
            },

            example: {
              type: Type.STRING,
            },

            practiceQuestion: {
              type: Type.STRING,
            },
          },

          required: [
            'hint',
            'simpleExplanation',
            'example',
            'practiceQuestion',
          ],
        },
      },
    })
  )

  const text =
    response.text?.trim() || '{}'

  const parsed =
    JSON.parse(text) as TutorResult

  if (
    !parsed.hint ||
    !parsed.simpleExplanation
  ) {
    throw new Error(
      'Invalid tutor response format from Gemini API'
    )
  }

  return parsed
}