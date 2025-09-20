import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelPerformanceTracker:
    """Sistema de tracking e métricas de performance dos modelos"""
    
    def __init__(self):
        self.predictions_history = []
        self.model_stats = defaultdict(dict)
        self.benchmark_results = {}
        
    def track_prediction(self, model_name: str, text: str, prediction: Dict[str, Any], 
                        execution_time: float, true_label: Optional[str] = None):
        """Registra uma predição para análise posterior"""
        record = {
            'timestamp': pd.Timestamp.now(),
            'model': model_name,
            'text': text[:100],  # Truncar para privacy
            'text_length': len(text),
            'predicted_label': prediction.get('label'),
            'confidence': prediction.get('score', 0),
            'execution_time': execution_time,
            'true_label': true_label,
            'correct': true_label == prediction.get('label') if true_label else None
        }
        self.predictions_history.append(record)
    
    def evaluate_model_accuracy(self, model_name: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Avalia accuracy de um modelo específico"""
        predictions = []
        true_labels = []
        execution_times = []
        
        for item in test_data:
            start_time = time.time()
            # Aqui seria feita a predição real
            pred_label = item.get('predicted_label')
            execution_time = time.time() - start_time
            
            predictions.append(pred_label)
            true_labels.append(item['true_label'])
            execution_times.append(execution_time)
        
        # Calcular métricas
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        cm = confusion_matrix(true_labels, predictions, labels=['negative', 'neutral', 'positive'])
        
        return {
            'model': model_name,
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'avg_execution_time': round(np.mean(execution_times), 4),
            'total_predictions': len(predictions),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Gera relatório completo de performance"""
        if not self.predictions_history:
            return {'error': 'Nenhuma predição registrada'}
        
        df = pd.DataFrame(self.predictions_history)
        
        # Estatísticas gerais
        report = {
            'summary': {
                'total_predictions': len(df),
                'unique_models': df['model'].nunique(),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'avg_confidence': round(df['confidence'].mean(), 3),
                'avg_execution_time': round(df['execution_time'].mean(), 4)
            },
            'by_model': {},
            'accuracy_metrics': {}
        }
        
        # Estatísticas por modelo
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            report['by_model'][model] = {
                'predictions': len(model_df),
                'avg_confidence': round(model_df['confidence'].mean(), 3),
                'avg_execution_time': round(model_df['execution_time'].mean(), 4),
                'label_distribution': model_df['predicted_label'].value_counts().to_dict()
            }
            
            # Se temos labels verdadeiros, calcular accuracy
            if model_df['correct'].notna().any():
                correct_predictions = model_df['correct'].sum()
                total_with_labels = model_df['correct'].notna().sum()
                accuracy = correct_predictions / total_with_labels if total_with_labels > 0 else 0
                
                report['accuracy_metrics'][model] = {
                    'accuracy': round(accuracy, 4),
                    'correct_predictions': int(correct_predictions),
                    'total_labeled': int(total_with_labels)
                }
        
        return report

class BenchmarkSuite:
    """Suite de testes benchmark para avaliação de modelos"""
    
    def __init__(self):
        self.benchmark_texts = self._create_benchmark_dataset()
    
    def _create_benchmark_dataset(self) -> List[Dict[str, Any]]:
        """Cria dataset de benchmark com casos de teste específicos"""
        return [
            # Casos claros positivos
            {'text': 'Amo muito este produto, é fantástico!', 'expected': 'positive', 'difficulty': 'easy'},
            {'text': 'Experiência incrível, superou expectativas!', 'expected': 'positive', 'difficulty': 'easy'},
            
            # Casos claros negativos  
            {'text': 'Péssimo produto, não funciona!', 'expected': 'negative', 'difficulty': 'easy'},
            {'text': 'Terrível, dinheiro jogado fora!', 'expected': 'negative', 'difficulty': 'easy'},
            
            # Casos neutros
            {'text': 'Produto normal, nada demais', 'expected': 'neutral', 'difficulty': 'medium'},
            {'text': 'OK, funciona como esperado', 'expected': 'neutral', 'difficulty': 'medium'},
            
            # Casos difíceis - negação
            {'text': 'Não é ruim, funciona bem', 'expected': 'positive', 'difficulty': 'hard'},
            {'text': 'Não gostei nada, muito ruim', 'expected': 'negative', 'difficulty': 'hard'},
            
            # Casos difíceis - sarcasmo
            {'text': 'Que maravilha... quebrou no primeiro dia', 'expected': 'negative', 'difficulty': 'hard'},
            {'text': 'Excelente qualidade... para o lixo', 'expected': 'negative', 'difficulty': 'hard'},
            
            # Casos com gírias brasileiras
            {'text': 'Bagulho massa demais, véi!', 'expected': 'positive', 'difficulty': 'medium'},
            {'text': 'Tá louco, que coisa ruim!', 'expected': 'negative', 'difficulty': 'medium'},
            
            # Casos com emojis
            {'text': 'Adorei! ❤️😍', 'expected': 'positive', 'difficulty': 'easy'},
            {'text': 'Que raiva 😡💔', 'expected': 'negative', 'difficulty': 'easy'},
        ]
    
    def run_benchmark(self, models: List[Any]) -> Dict[str, Any]:
        """Executa benchmark em todos os modelos"""
        results = {}
        
        for model in models:
            model_name = getattr(model, '__class__', type(model)).__name__
            results[model_name] = self._benchmark_single_model(model)
        
        return {
            'benchmark_results': results,
            'comparison': self._compare_models(results),
            'summary': self._generate_benchmark_summary(results)
        }
    
    def _benchmark_single_model(self, model) -> Dict[str, Any]:
        """Benchmark para um modelo específico"""
        results = []
        total_time = 0
        
        for test_case in self.benchmark_texts:
            start_time = time.time()
            
            try:
                prediction = model.analyze(test_case['text'])
                execution_time = time.time() - start_time
                total_time += execution_time
                
                is_correct = prediction['label'].lower() == test_case['expected'].lower()
                
                results.append({
                    'text': test_case['text'],
                    'expected': test_case['expected'],
                    'predicted': prediction['label'],
                    'confidence': prediction.get('score', 0),
                    'correct': is_correct,
                    'difficulty': test_case['difficulty'],
                    'execution_time': execution_time
                })
                
            except Exception as e:
                results.append({
                    'text': test_case['text'],
                    'expected': test_case['expected'],
                    'predicted': 'ERROR',
                    'confidence': 0,
                    'correct': False,
                    'difficulty': test_case['difficulty'],
                    'execution_time': 0,
                    'error': str(e)
                })
        
        # Calcular estatísticas
        correct_predictions = sum(1 for r in results if r['correct'])
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Accuracy por dificuldade
        difficulty_stats = {}
        for difficulty in ['easy', 'medium', 'hard']:
            diff_results = [r for r in results if r['difficulty'] == difficulty]
            if diff_results:
                correct = sum(1 for r in diff_results if r['correct'])
                total = len(diff_results)
                difficulty_stats[difficulty] = {
                    'accuracy': round(correct / total, 4) if total > 0 else 0,
                    'correct': correct,
                    'total': total
                }
        
        return {
            'overall_accuracy': round(accuracy, 4),
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'avg_confidence': round(np.mean([r['confidence'] for r in results]), 3),
            'total_execution_time': round(total_time, 4),
            'avg_execution_time': round(total_time / total_predictions, 4),
            'difficulty_breakdown': difficulty_stats,
            'detailed_results': results
        }
    
    def _compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compara performance entre modelos"""
        if len(results) < 2:
            return {}
        
        comparison = {
            'best_accuracy': max(results.items(), key=lambda x: x[1]['overall_accuracy']),
            'fastest_model': min(results.items(), key=lambda x: x[1]['avg_execution_time']),
            'most_confident': max(results.items(), key=lambda x: x[1]['avg_confidence']),
            'accuracy_ranking': sorted(results.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True),
            'speed_ranking': sorted(results.items(), key=lambda x: x[1]['avg_execution_time'])
        }
        
        return comparison
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera sumário do benchmark"""
        if not results:
            return {}
        
        accuracies = [r['overall_accuracy'] for r in results.values()]
        times = [r['avg_execution_time'] for r in results.values()]
        
        return {
            'total_models_tested': len(results),
            'accuracy_stats': {
                'mean': round(np.mean(accuracies), 4),
                'std': round(np.std(accuracies), 4),
                'min': round(min(accuracies), 4),
                'max': round(max(accuracies), 4)
            },
            'execution_time_stats': {
                'mean': round(np.mean(times), 4),
                'std': round(np.std(times), 4),
                'min': round(min(times), 4),
                'max': round(max(times), 4)
            },
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas no benchmark"""
        recommendations = []
        
        # Encontrar melhor modelo geral
        best_model = max(results.items(), key=lambda x: x[1]['overall_accuracy'])
        recommendations.append(f"Melhor accuracy geral: {best_model[0]} ({best_model[1]['overall_accuracy']:.1%})")
        
        # Modelo mais rápido com boa accuracy
        good_accuracy_models = [(k, v) for k, v in results.items() if v['overall_accuracy'] > 0.7]
        if good_accuracy_models:
            fastest_good = min(good_accuracy_models, key=lambda x: x[1]['avg_execution_time'])
            recommendations.append(f"Melhor custo-benefício: {fastest_good[0]} (accuracy: {fastest_good[1]['overall_accuracy']:.1%}, tempo: {fastest_good[1]['avg_execution_time']:.3f}s)")
        
        # Verificar performance em casos difíceis
        for model_name, model_results in results.items():
            hard_accuracy = model_results.get('difficulty_breakdown', {}).get('hard', {}).get('accuracy', 0)
            if hard_accuracy > 0.8:
                recommendations.append(f"{model_name} tem excelente performance em casos difíceis ({hard_accuracy:.1%})")
        
        return recommendations

class RealTimeMonitor:
    """Monitor em tempo real de performance do sistema"""
    
    def __init__(self):
        self.current_stats = {
            'requests_per_minute': 0,
            'avg_response_time': 0,
            'error_rate': 0,
            'active_models': set()
        }
        self.recent_requests = []
    
    def log_request(self, model_name: str, execution_time: float, success: bool):
        """Registra uma requisição para monitoramento"""
        now = pd.Timestamp.now()
        
        self.recent_requests.append({
            'timestamp': now,
            'model': model_name,
            'execution_time': execution_time,
            'success': success
        })
        
        # Limpar requests antigos (últimos 5 minutos)
        cutoff = now - pd.Timedelta(minutes=5)
        self.recent_requests = [r for r in self.recent_requests if r['timestamp'] > cutoff]
        
        self._update_current_stats()
    
    def _update_current_stats(self):
        """Atualiza estatísticas atuais"""
        if not self.recent_requests:
            return
        
        now = pd.Timestamp.now()
        last_minute = now - pd.Timedelta(minutes=1)
        recent = [r for r in self.recent_requests if r['timestamp'] > last_minute]
        
        self.current_stats = {
            'requests_per_minute': len(recent),
            'avg_response_time': round(np.mean([r['execution_time'] for r in recent]), 4) if recent else 0,
            'error_rate': round(sum(1 for r in recent if not r['success']) / len(recent), 3) if recent else 0,
            'active_models': set(r['model'] for r in recent)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Retorna status de saúde do sistema"""
        self._update_current_stats()
        
        # Determinar status geral
        status = 'healthy'
        issues = []
        
        if self.current_stats['error_rate'] > 0.05:  # >5% erro
            status = 'degraded'
            issues.append(f"Alta taxa de erro: {self.current_stats['error_rate']:.1%}")
        
        if self.current_stats['avg_response_time'] > 2.0:  # >2s resposta
            status = 'slow' if status == 'healthy' else 'critical'
            issues.append(f"Tempo de resposta alto: {self.current_stats['avg_response_time']:.2f}s")
        
        return {
            'status': status,
            'timestamp': pd.Timestamp.now().isoformat(),
            'current_stats': self.current_stats,
            'issues': issues,
            'uptime_info': {
                'total_requests_5min': len(self.recent_requests),
                'successful_requests': sum(1 for r in self.recent_requests if r['success']),
                'active_models': list(self.current_stats['active_models'])
            }
        }
