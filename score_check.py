from flask import Flask, request, jsonify
import re
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class QueryScore:
    complexity_score: float
    urgency_score: float
    domain_specificity: float
    context_length_score: float
    legal_terminology_score: float
    precedent_requirement_score: float
    jurisdiction_complexity_score: float
    total_score: float
    recommended_model: str
    confidence: float
    reasoning: List[str]

class LegalQueryScorer:
    def __init__(self):
        # Load sentence transformer for semantic analysis
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Legal domain keywords and their complexity weights
        self.legal_domains = {
            'criminal': {
                'keywords': ['criminal', 'crime', 'arrest', 'prosecution', 'felony', 'misdemeanor', 
                           'bail', 'conviction', 'sentencing', 'plea', 'defendant', 'prosecutor'],
                'base_complexity': 7.5,
                'requires_precedent': True
            },
            'civil': {
                'keywords': ['contract', 'tort', 'negligence', 'damages', 'liability', 'breach', 
                           'plaintiff', 'defendant', 'settlement', 'judgment'],
                'base_complexity': 6.0,
                'requires_precedent': True
            },
            'family': {
                'keywords': ['divorce', 'custody', 'marriage', 'adoption', 'alimony', 'child support', 
                           'domestic', 'separation', 'prenup', 'visitation'],
                'base_complexity': 5.5,
                'requires_precedent': False
            },
            'corporate': {
                'keywords': ['corporation', 'business', 'merger', 'securities', 'compliance', 'shareholder', 
                           'board', 'fiduciary', 'intellectual property', 'patent', 'trademark'],
                'base_complexity': 8.0,
                'requires_precedent': True
            },
            'employment': {
                'keywords': ['employment', 'labor', 'discrimination', 'wrongful termination', 'harassment', 
                           'wage', 'overtime', 'worker', 'employer', 'union'],
                'base_complexity': 6.5,
                'requires_precedent': True
            },
            'property': {
                'keywords': ['property', 'real estate', 'lease', 'mortgage', 'zoning', 'easement', 
                           'deed', 'title', 'landlord', 'tenant'],
                'base_complexity': 5.0,
                'requires_precedent': False
            },
            'constitutional': {
                'keywords': ['constitution', 'amendment', 'rights', 'freedom', 'supreme court', 
                           'federal', 'constitutional', 'bill of rights'],
                'base_complexity': 9.5,
                'requires_precedent': True
            }
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'high': {
                'keywords': ['precedent', 'appeal', 'supreme court', 'constitutional', 'federal', 
                           'interstate', 'international', 'class action', 'injunction', 'statute of limitations'],
                'weight': 2.0
            },
            'medium': {
                'keywords': ['court', 'judge', 'lawyer', 'attorney', 'legal', 'law', 'statute', 
                           'regulation', 'violation', 'penalty'],
                'weight': 1.0
            },
            'procedural': {
                'keywords': ['file', 'filing', 'deadline', 'procedure', 'form', 'document', 
                           'paperwork', 'submit', 'application'],
                'weight': 0.5
            }
        }
        
        # Urgency indicators
        self.urgency_keywords = {
            'critical': ['emergency', 'urgent', 'immediate', 'asap', 'deadline today', 'court tomorrow'],
            'high': ['soon', 'quickly', 'deadline', 'time sensitive', 'hearing', 'court date'],
            'medium': ['when', 'how long', 'timeline', 'schedule'],
            'low': ['general', 'information', 'understand', 'explain', 'what is']
        }
        
        # Jurisdiction complexity indicators
        self.jurisdiction_indicators = {
            'international': ['international', 'treaty', 'foreign', 'cross-border', 'extradition'],
            'federal': ['federal', 'fbi', 'irs', 'sec', 'ftc', 'interstate'],
            'multi_state': ['multi-state', 'interstate', 'multiple states', 'different states'],
            'state': ['state law', 'state court', 'state regulation'],
            'local': ['city', 'county', 'municipal', 'local ordinance']
        }
        
        # Model selection thresholds
        self.thresholds = {
            'slm_max_score': 6.0,
            'llm_min_score': 6.5,
            'confidence_threshold': 0.7
        }
        
        # Pre-compute domain embeddings for faster classification
        self.domain_embeddings = self._precompute_domain_embeddings()
    
    def _precompute_domain_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for each legal domain"""
        domain_embeddings = {}
        
        if self.embedding_model is None:
            return domain_embeddings
        
        for domain, info in self.legal_domains.items():
            # Create a representative text for the domain
            domain_text = ' '.join(info['keywords'])
            try:
                embedding = self.embedding_model.encode([domain_text])[0]
                domain_embeddings[domain] = embedding
            except Exception as e:
                logger.error(f"Failed to compute embedding for domain {domain}: {e}")
        
        return domain_embeddings
    
    def analyze_query(self, query: str, user_context: Optional[Dict] = None) -> QueryScore:
        """Main method to analyze query and determine model selection"""
        start_time = time.time()
        
        # Normalize query
        normalized_query = self._normalize_query(query)
        
        # Calculate individual scores
        complexity_score = self._calculate_complexity_score(normalized_query)
        urgency_score = self._calculate_urgency_score(normalized_query)
        domain_specificity = self._calculate_domain_specificity(normalized_query)
        context_length_score = self._calculate_context_length_score(normalized_query)
        legal_terminology_score = self._calculate_legal_terminology_score(normalized_query)
        precedent_requirement_score = self._calculate_precedent_requirement_score(normalized_query)
        jurisdiction_complexity_score = self._calculate_jurisdiction_complexity_score(normalized_query)
        
        # Calculate weighted total score
        total_score = self._calculate_total_score(
            complexity_score, urgency_score, domain_specificity,
            context_length_score, legal_terminology_score,
            precedent_requirement_score, jurisdiction_complexity_score
        )
        
        # Determine recommended model
        recommended_model, confidence, reasoning = self._determine_model_selection(
            total_score, complexity_score, urgency_score, domain_specificity,
            precedent_requirement_score, normalized_query, user_context
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Query analysis completed in {processing_time:.3f}s")
        
        return QueryScore(
            complexity_score=complexity_score,
            urgency_score=urgency_score,
            domain_specificity=domain_specificity,
            context_length_score=context_length_score,
            legal_terminology_score=legal_terminology_score,
            precedent_requirement_score=precedent_requirement_score,
            jurisdiction_complexity_score=jurisdiction_complexity_score,
            total_score=total_score,
            recommended_model=recommended_model,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent analysis"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Expand common abbreviations
        abbreviations = {
            r'\bdui\b': 'driving under influence',
            r'\bdwi\b': 'driving while intoxicated',
            r'\bllc\b': 'limited liability company',
            r'\binc\b': 'incorporated',
            r'\bcorp\b': 'corporation',
            r'\batty\b': 'attorney',
            r'\blaw\s+suit\b': 'lawsuit',
            r'\bco\.\b': 'company'
        }
        
        for abbrev, expansion in abbreviations.items():
            query = re.sub(abbrev, expansion, query)
        
        return query
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Calculate complexity based on legal concepts and terminology"""
        score = 0.0
        
        # Base complexity from query length and structure
        word_count = len(query.split())
        if word_count > 50:
            score += 2.0
        elif word_count > 20:
            score += 1.0
        elif word_count < 5:
            score -= 1.0
        
        # Complexity from legal indicators
        for level, info in self.complexity_indicators.items():
            matches = sum(1 for keyword in info['keywords'] if keyword in query)
            score += matches * info['weight']
        
        # Multiple legal concepts increase complexity
        legal_concept_count = 0
        legal_concepts = ['contract', 'tort', 'criminal', 'constitutional', 'statutory', 'regulatory']
        for concept in legal_concepts:
            if concept in query:
                legal_concept_count += 1
        
        if legal_concept_count > 2:
            score += 2.0
        elif legal_concept_count > 1:
            score += 1.0
        
        # Question complexity indicators
        complex_question_patterns = [
            r'what are the implications of',
            r'how does .* affect',
            r'what happens if .* and .*',
            r'compare .* with .*',
            r'analyze the relationship between'
        ]
        
        for pattern in complex_question_patterns:
            if re.search(pattern, query):
                score += 1.5
                break
        
        return min(score, 10.0)
    
    def _calculate_urgency_score(self, query: str) -> float:
        """Calculate urgency based on time-sensitive keywords"""
        score = 0.0
        
        for urgency_level, keywords in self.urgency_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query)
            if matches > 0:
                if urgency_level == 'critical':
                    score += matches * 3.0
                elif urgency_level == 'high':
                    score += matches * 2.0
                elif urgency_level == 'medium':
                    score += matches * 1.0
                else:  # low
                    score -= matches * 0.5
        
        # Time-specific patterns
        time_patterns = [
            r'\b(today|tomorrow|this week)\b',
            r'\b\d+\s+(days?|weeks?|months?)\b',
            r'\bdeadline\s+in\s+\d+',
            r'\bcourt\s+date\s+(today|tomorrow|this week)'
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, query):
                score += 2.0
        
        return min(score, 10.0)
    
    def _calculate_domain_specificity(self, query: str) -> float:
        """Calculate how specific the query is to particular legal domains"""
        domain_scores = {}
        
        # Keyword-based domain detection
        for domain, info in self.legal_domains.items():
            matches = sum(1 for keyword in info['keywords'] if keyword in query)
            if matches > 0:
                domain_scores[domain] = matches * info['base_complexity']
        
        # Semantic similarity if embedding model is available
        if self.embedding_model and self.domain_embeddings:
            try:
                query_embedding = self.embedding_model.encode([query])[0]
                
                for domain, domain_embedding in self.domain_embeddings.items():
                    similarity = np.dot(query_embedding, domain_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(domain_embedding)
                    )
                    
                    # Weight semantic similarity
                    semantic_score = similarity * 5.0
                    
                    if domain in domain_scores:
                        domain_scores[domain] = max(domain_scores[domain], semantic_score)
                    else:
                        domain_scores[domain] = semantic_score
                        
            except Exception as e:
                logger.error(f"Error in semantic similarity calculation: {e}")
        
        # Return the highest domain specificity score
        if domain_scores:
            max_score = max(domain_scores.values())
            return min(max_score, 10.0)
        
        return 2.0  # Default score for general queries
    
    def _calculate_context_length_score(self, query: str) -> float:
        """Calculate score based on context length requirements"""
        word_count = len(query.split())
        
        # Longer queries typically need more sophisticated processing
        if word_count > 100:
            return 8.0
        elif word_count > 50:
            return 6.0
        elif word_count > 20:
            return 4.0
        elif word_count > 10:
            return 2.0
        else:
            return 1.0
    
    def _calculate_legal_terminology_score(self, query: str) -> float:
        """Calculate score based on legal terminology complexity"""
        advanced_legal_terms = [
            'estoppel', 'indemnification', 'subrogation', 'novation', 'rescission',
            'restitution', 'quantum meruit', 'res ipsa loquitur', 'stare decisis',
            'habeas corpus', 'mens rea', 'actus reus', 'voir dire', 'mandamus',
            'certiorari', 'enjoin', 'pleadings', 'discovery', 'deposition'
        ]
        
        basic_legal_terms = [
            'contract', 'agreement', 'lawsuit', 'court', 'judge', 'lawyer',
            'attorney', 'legal', 'law', 'rights', 'liable', 'damages'
        ]
        
        score = 0.0
        
        # Advanced terms significantly increase complexity
        advanced_matches = sum(1 for term in advanced_legal_terms if term in query)
        score += advanced_matches * 2.0
        
        # Basic terms provide moderate complexity
        basic_matches = sum(1 for term in basic_legal_terms if term in query)
        score += basic_matches * 0.5
        
        return min(score, 10.0)
    
    def _calculate_precedent_requirement_score(self, query: str) -> float:
        """Calculate score based on need for legal precedents"""
        precedent_indicators = [
            'precedent', 'case law', 'court decision', 'ruling', 'judgment',
            'similar case', 'what happened in', 'court held', 'decided',
            'interpretation', 'how courts', 'judicial', 'appeal'
        ]
        
        score = 0.0
        matches = sum(1 for indicator in precedent_indicators if indicator in query)
        score += matches * 1.5
        
        # Questions about specific legal outcomes often need precedent analysis
        outcome_patterns = [
            r'what happens if',
            r'what would happen',
            r'likely outcome',
            r'chances of winning',
            r'court decide',
            r'judge rule'
        ]
        
        for pattern in outcome_patterns:
            if re.search(pattern, query):
                score += 2.0
        
        return min(score, 10.0)
    
    def _calculate_jurisdiction_complexity_score(self, query: str) -> float:
        """Calculate complexity based on jurisdictional issues"""
        score = 0.0
        
        for jurisdiction_level, keywords in self.jurisdiction_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in query)
            if matches > 0:
                if jurisdiction_level == 'international':
                    score += matches * 3.0
                elif jurisdiction_level == 'federal':
                    score += matches * 2.5
                elif jurisdiction_level == 'multi_state':
                    score += matches * 2.0
                elif jurisdiction_level == 'state':
                    score += matches * 1.0
                else:  # local
                    score += matches * 0.5
        
        # Multiple jurisdiction indicators
        jurisdiction_count = sum(1 for keywords in self.jurisdiction_indicators.values() 
                               for keyword in keywords if keyword in query)
        if jurisdiction_count > 2:
            score += 1.5
        
        return min(score, 10.0)
    
    def _calculate_total_score(self, complexity_score: float, urgency_score: float,
                             domain_specificity: float, context_length_score: float,
                             legal_terminology_score: float, precedent_requirement_score: float,
                             jurisdiction_complexity_score: float) -> float:
        """Calculate weighted total score"""
        
        # Weights for different components
        weights = {
            'complexity': 0.25,
            'urgency': 0.15,
            'domain_specificity': 0.20,
            'context_length': 0.10,
            'legal_terminology': 0.15,
            'precedent_requirement': 0.10,
            'jurisdiction_complexity': 0.05
        }
        
        total_score = (
            complexity_score * weights['complexity'] +
            urgency_score * weights['urgency'] +
            domain_specificity * weights['domain_specificity'] +
            context_length_score * weights['context_length'] +
            legal_terminology_score * weights['legal_terminology'] +
            precedent_requirement_score * weights['precedent_requirement'] +
            jurisdiction_complexity_score * weights['jurisdiction_complexity']
        )
        
        return min(total_score, 10.0)
    
    def _determine_model_selection(self, total_score: float, complexity_score: float,
                                 urgency_score: float, domain_specificity: float,
                                 precedent_requirement_score: float, query: str,
                                 user_context: Optional[Dict] = None) -> Tuple[str, float, List[str]]:
        """Determine which model to use based on scores and additional factors"""
        
        reasoning = []
        confidence = 0.7  # Base confidence
        
        # Primary decision based on total score
        if total_score >= 7.5:
            model = 'llm'
            reasoning.append(f"High total complexity score ({total_score:.1f}/10.0)")
            confidence += 0.2
        elif total_score <= 4.0:
            model = 'slm'
            reasoning.append(f"Low total complexity score ({total_score:.1f}/10.0)")
            confidence += 0.15
        else:
            # Additional decision factors for middle range
            model = 'slm'  # Default to SLM for efficiency
            
            # Override conditions for LLM
            if complexity_score > 7.0:
                model = 'llm'
                reasoning.append(f"High complexity score ({complexity_score:.1f}/10.0)")
                confidence += 0.1
            
            if precedent_requirement_score > 5.0:
                model = 'llm'
                reasoning.append(f"Requires precedent analysis ({precedent_requirement_score:.1f}/10.0)")
                confidence += 0.1
            
            if domain_specificity > 8.0:
                model = 'llm'
                reasoning.append(f"Highly domain-specific query ({domain_specificity:.1f}/10.0)")
                confidence += 0.1
            
            # Urgency can override to SLM for speed
            if urgency_score > 8.0 and model == 'llm':
                model = 'slm'
                reasoning.append(f"High urgency overrides complexity for speed ({urgency_score:.1f}/10.0)")
                confidence -= 0.05
        
        # User context considerations
        if user_context:
            if user_context.get('user_type') == 'lawyer' and model == 'slm':
                model = 'llm'
                reasoning.append("Professional user requires comprehensive analysis")
                confidence += 0.1
            
            if user_context.get('session_history', {}).get('complex_queries', 0) > 3:
                if model == 'slm':
                    model = 'llm'
                    reasoning.append("User pattern indicates need for detailed analysis")
                    confidence += 0.05
        
        # Final adjustments
        confidence = min(confidence, 0.95)
        confidence = max(confidence, 0.5)
        
        if not reasoning:
            reasoning.append(f"Standard decision based on total score ({total_score:.1f}/10.0)")
        
        return model, confidence, reasoning

# Initialize the scorer
scorer = LegalQueryScorer()

# Flask Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'embedding_model_loaded': scorer.embedding_model is not None
    })

@app.route('/score_query', methods=['POST'])
def score_query():
    """Main endpoint to score a legal query and get model recommendation"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query in request body',
                'status': 'error'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'error': 'Query cannot be empty',
                'status': 'error'
            }), 400
        
        user_context = data.get('user_context', {})
        
        # Analyze the query
        result = scorer.analyze_query(query, user_context)
        
        # Convert to dictionary for JSON response
        response = asdict(result)
        response['status'] = 'success'
        response['timestamp'] = datetime.now().isoformat()
        response['query_hash'] = hashlib.md5(query.encode()).hexdigest()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in score_query: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_score', methods=['POST'])
def batch_score_queries():
    """Endpoint to score multiple queries at once"""
    try:
        data = request.get_json()
        
        if not data or 'queries' not in data:
            return jsonify({
                'error': 'Missing queries array in request body',
                'status': 'error'
            }), 400
        
        queries = data['queries']
        if not isinstance(queries, list) or len(queries) == 0:
            return jsonify({
                'error': 'Queries must be a non-empty array',
                'status': 'error'
            }), 400
        
        if len(queries) > 50:  # Limit batch size
            return jsonify({
                'error': 'Maximum 50 queries allowed per batch',
                'status': 'error'
            }), 400
        
        user_context = data.get('user_context', {})
        results = []
        
        for i, query in enumerate(queries):
            if not isinstance(query, str) or not query.strip():
                results.append({
                    'index': i,
                    'error': 'Invalid query at index {}'.format(i),
                    'status': 'error'
                })
                continue
            
            try:
                result = scorer.analyze_query(query.strip(), user_context)
                response = asdict(result)
                response['index'] = i
                response['status'] = 'success'
                response['query_hash'] = hashlib.md5(query.encode()).hexdigest()
                results.append(response)
            except Exception as e:
                results.append({
                    'index': i,
                    'error': f'Error processing query: {str(e)}',
                    'status': 'error'
                })
        
        return jsonify({
            'results': results,
            'total_queries': len(queries),
            'successful': sum(1 for r in results if r.get('status') == 'success'),
            'failed': sum(1 for r in results if r.get('status') == 'error'),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in batch_score_queries: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model_stats', methods=['GET'])
def get_model_stats():
    """Get statistics about model selection patterns"""
    # This would typically come from a database in production
    # For now, return mock statistics
    return jsonify({
        'total_queries_analyzed': 1250,
        'slm_selected': 875,
        'llm_selected': 375,
        'slm_percentage': 70.0,
        'llm_percentage': 30.0,
        'average_confidence': 0.82,
        'top_domains': [
            {'domain': 'civil', 'count': 340, 'avg_score': 6.2},
            {'domain': 'employment', 'count': 280, 'avg_score': 6.8},
            {'domain': 'family', 'count': 220, 'avg_score': 5.1},
            {'domain': 'criminal', 'count': 185, 'avg_score': 7.5},
            {'domain': 'corporate', 'count': 125, 'avg_score': 8.1}
        ],
        'performance_metrics': {
            'avg_processing_time_ms': 45,
            'cache_hit_rate': 0.63,
            'accuracy_score': 0.89
        },
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    })

@app.route('/explain_score', methods=['POST'])
def explain_score():
    """Detailed explanation of how a score was calculated"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query in request body',
                'status': 'error'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'error': 'Query cannot be empty',
                'status': 'error'
            }), 400
        
        # Analyze the query
        result = scorer.analyze_query(query, data.get('user_context', {}))
        
        # Create detailed explanation
        explanation = {
            'query': query,
            'normalized_query': scorer._normalize_query(query),
            'final_recommendation': result.recommended_model,
            'total_score': result.total_score,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'score_breakdown': {
                'complexity_score': {
                    'value': result.complexity_score,
                    'weight': 0.25,
                    'contribution': result.complexity_score * 0.25,
                    'description': 'Based on legal concepts, terminology, and query structure'
                },
                'urgency_score': {
                    'value': result.urgency_score,
                    'weight': 0.15,
                    'contribution': result.urgency_score * 0.15,
                    'description': 'Based on time-sensitive keywords and deadlines'
                },
                'domain_specificity': {
                    'value': result.domain_specificity,
                    'weight': 0.20,
                    'contribution': result.domain_specificity * 0.20,
                    'description': 'How specific the query is to particular legal domains'
                },
                'context_length_score': {
                    'value': result.context_length_score,
                    'weight': 0.10,
                    'contribution': result.context_length_score * 0.10,
                    'description': 'Based on query length and context requirements'
                },
                'legal_terminology_score': {
                    'value': result.legal_terminology_score,
                    'weight': 0.15,
                    'contribution': result.legal_terminology_score * 0.15,
                    'description': 'Based on complexity of legal terms used'
                },
                'precedent_requirement_score': {
                    'value': result.precedent_requirement_score,
                    'weight': 0.10,
                    'contribution': result.precedent_requirement_score * 0.10,
                    'description': 'Whether the query requires case law analysis'
                },
                'jurisdiction_complexity_score': {
                    'value': result.jurisdiction_complexity_score,
                    'weight': 0.05,
                    'contribution': result.jurisdiction_complexity_score * 0.05,
                    'description': 'Complexity based on jurisdictional issues'
                }
            },
            'decision_thresholds': {
                'slm_recommended_below': scorer.thresholds['slm_max_score'],
                'llm_recommended_above': scorer.thresholds['llm_min_score'],
                'confidence_threshold': scorer.thresholds['confidence_threshold']
            },
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(explanation)
        
    except Exception as e:
        logger.error(f"Error in explain_score: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # You can set host and port here.
    # For development, host='0.0.0.0' makes it accessible from other machines on the network.
    # For production, it's recommended to run with a WSGI server like Gunicorn.
    app.run(debug=True, host='0.0.0.0', port=5000)