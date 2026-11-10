"""离线回归测试：覆盖数据适配、评分、请求协议和结果统计，不占用 GPU。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from scripts.evaluation import data_utils, metrics, model_client, prompts, runner
from scripts.evaluation.config import CMSQA_LABELS, PAWS_LABELS, SST5_LABELS
from scripts.evaluation.task_data import TaskData
from scripts.example_selectors import (
    BM25Selector, CEILSelector, GenICLSelector, KATESelector, greedy_dpp_map,
    latent_description,
    select_random_examples,
)


class FakeDenseEncoder:
    """测试用确定性 encoder；避免下载模型或占用 GPU。"""

    cache_key = "fake-dense-encoder-v1"

    def __init__(self):
        self.vectors = {
            "cash withdrawal from an ATM": [1.0, 0.0, 0.0],
            "my new card has not arrived": [0.0, 1.0, 0.0],
            "international bank transfer": [0.0, 0.0, 1.0],
            "where is my card delivery": [0.0, 0.9, 0.1],
        }

    def encode(self, texts):
        import numpy as np

        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class FakeGenICLEncoder:
    cache_key = "fake-genicl-e5"

    def encode(self, texts):
        import numpy as np

        vectors = {
            "alpha\nyes": [1.0, 0.0],
            "beta\nno": [0.8, 0.2],
            "gamma\nyes": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


class FakeGenICLScorer:
    def score(self, prompts, completions):
        import numpy as np

        # 最终排序必须由公式（11）决定，而不是由 E5 相似度决定。
        return np.asarray([1.0 if prompt.startswith("beta") else 0.1 for prompt in prompts])


class EvaluationTests(unittest.TestCase):
    """验证拆分后各模块之间的关键契约。"""

    def test_label_scoring(self):
        # 长标签不能被内部短标签覆盖；多个不同标签判无效。
        self.assertEqual(metrics.parse_prediction('Label: very negative.', SST5_LABELS), 'very negative')
        self.assertIsNone(metrics.parse_prediction('positive or negative', SST5_LABELS))
        self.assertEqual(metrics.parse_prediction('Answer: B', CMSQA_LABELS), 'B')
        self.assertFalse(metrics.exact_match(None, 'yes'))

    def test_dataset_adapters(self):
        # 用本地小样本替代远端下载，核对四种 Hugging Face 数据格式。
        cmsqa = {
            'question': 'Where would a fish live?',
            'choices': {
                'label': ['A', 'B', 'C', 'D', 'E'],
                'text': ['tree', 'water', 'road', 'desk', 'roof'],
            },
            'answerKey': 'B',
        }
        fixtures = [
            (data_utils.load_sst5, {'text': 'good', 'label': 4}, 'very positive', 'label'),
            (
                data_utils.load_paws,
                {'sentence1': 'A cat sleeps.', 'sentence2': 'A feline is sleeping.', 'label': 1},
                'paraphrase', 'label',
            ),
            (data_utils.load_boolq, {'passage': 'A', 'question': 'B?', 'answer': False}, 'no', 'label'),
            (data_utils.load_cmsqa, cmsqa, 'B', 'label'),
        ]
        for loader, example, gold, answer_type in fixtures:
            with patch.object(data_utils, 'load_dataset', return_value=[example]):
                task = loader(cache_dir=Path('/unused'))
            self.assertEqual(task.samples[0]['gold'], gold)
            self.assertEqual(task.answer_type, answer_type)
        self.assertEqual(task.samples[0]['input']['choices'][1]['text'], 'water')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Banking77 标签按编号排列，与文件中顺序无关。
            (root / 'labels.json').write_text(json.dumps([
                {'label_id': 1, 'label': 'b'}, {'label_id': 0, 'label': 'a'}
            ]))
            (root / 'test.jsonl').write_text(json.dumps({'text': 'hello', 'label': 'a'}) + '\n')
            (root / 'train.jsonl').write_text(
                json.dumps({'id': 'train_0', 'text': 'example', 'label': 'b'}) + '\n'
            )
            task = data_utils.load_banking77(banking77_dir=root)
            self.assertEqual(task.labels, ('a', 'b'))
            task = data_utils.load_banking77(banking77_dir=root, load_demonstrations=True)
            self.assertEqual(task.demonstration_pool[0]['id'], 'train_0')
            self.assertEqual(task.demonstration_pool[0]['gold'], 'b')

    def test_demonstration_pool_removes_contamination_and_duplicates(self):
        samples = [
            {'input': {'text': '  SAME   Text '}, 'gold': 'a'},
            {
                'input': {'passage': 'Shared passage', 'question': 'Test question?'},
                'gold': 'yes',
            },
        ]
        pool = [
            # 大小写和空白不同，但规范化后与测试输入相同，必须删除。
            {'id': 'overlap', 'input': {'text': 'same text'}, 'gold': 'a'},
            # 相同 passage、不同 question 是不同的完整输入，必须保留。
            {
                'id': 'boolq_distinct',
                'input': {'passage': 'Shared passage', 'question': 'Other question?'},
                'gold': 'no',
            },
            # 同标签的候选池内部重复只保留第一条。
            {'id': 'duplicate_first', 'input': {'text': 'Keep me'}, 'gold': 'b'},
            {'id': 'duplicate_second', 'input': {'text': ' keep  me '}, 'gold': 'b'},
            # 同一输入标签冲突时整组排除。
            {'id': 'conflict_a', 'input': {'text': 'Conflict'}, 'gold': 'a'},
            {'id': 'conflict_b', 'input': {'text': 'conflict'}, 'gold': 'b'},
        ]

        cleaned, stats = data_utils.sanitize_demonstration_pool(samples, pool)

        self.assertEqual(
            [candidate['id'] for candidate in cleaned],
            ['boolq_distinct', 'duplicate_first'],
        )
        self.assertEqual(stats['original_pool_size'], 6)
        self.assertEqual(stats['test_overlap_rows_removed'], 1)
        self.assertEqual(stats['test_overlap_label_conflicts'], 0)
        self.assertEqual(stats['internal_duplicate_rows_removed'], 1)
        self.assertEqual(stats['conflicting_rows_removed'], 2)
        self.assertEqual(stats['final_pool_size'], 2)

    def test_client_retry_and_payload(self):
        # 一次临时错误后成功，核对重试次数和分类输出约束。
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': 'yes'}}]}
        with patch.object(
            model_client._local_session,
            'post',
            side_effect=[requests.ConnectionError(), response],
        ) as post:
            with patch.object(model_client.time, 'sleep'):
                result = model_client.query_vllm(
                    'question', ('yes', 'no'), 'http://127.0.0.1:8002/',
                    'model', 1, 2, True,
                )
        self.assertEqual(result, 'yes')
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs['json']['structured_outputs'], {'choice': ['yes', 'no']})
        self.assertEqual(post.call_args.kwargs['json']['max_tokens'], 32)

    def test_local_vllm_bypasses_environment_proxy(self):
        self.assertIs(
            model_client._http_client('http://127.0.0.1:8002'),
            model_client._local_session,
        )
        self.assertIs(
            model_client._http_client('http://localhost:8001'),
            model_client._local_session,
        )
        self.assertFalse(model_client._local_session.trust_env)
        self.assertIs(
            model_client._http_client('https://models.example.com'),
            model_client.requests,
        )

    def test_paws_and_cmsqa_icl_prompts_match_their_tasks(self):
        paws_prompt = prompts.build_paws_prompt('A dog runs.', 'A canine is running.')
        paws_demo = [{
            'id': 'p1',
            'input': {'sentence1': 'A cat sleeps.', 'sentence2': 'A feline sleeps.'},
            'gold': 'paraphrase',
        }]
        paws_icl = prompts.build_icl_prompt('paws', paws_prompt, paws_demo)
        self.assertIn('Sentence 1:\nA cat sleeps.', paws_icl)
        self.assertIn('Sentence 2:\nA feline sleeps.', paws_icl)
        self.assertGreater(paws_icl.rfind('A dog runs.'), paws_icl.rfind('Example 1:'))

        choices = [
            {'label': label, 'text': text}
            for label, text in zip(CMSQA_LABELS, ('tree', 'water', 'road', 'desk', 'roof'))
        ]
        cmsqa_prompt = prompts.build_cmsqa_prompt('Where does a fish live?', choices)
        cmsqa_demo = [{
            'id': 'c1', 'input': {'question': 'Where does a bird fly?', 'choices': choices},
            'gold': 'A',
        }]
        cmsqa_icl = prompts.build_icl_prompt('cmsqa', cmsqa_prompt, cmsqa_demo)
        self.assertIn('Options:\nA. tree\nB. water', cmsqa_icl)
        self.assertGreater(cmsqa_icl.rfind('Where does a fish live?'), cmsqa_icl.rfind('Example 1:'))

    def test_nested_inputs_are_available_to_retrievers(self):
        text = __import__('scripts.example_selectors.bm25_selector', fromlist=['input_to_text']).input_to_text({
            'question': 'Where does a fish live?',
            'choices': [{'label': 'A', 'text': 'tree'}, {'label': 'B', 'text': 'water'}],
        })
        self.assertIn('fish', text)
        self.assertIn('water', text)

    def test_genicl_uses_e5_subset_then_latent_probability_reranking(self):
        pool = [
            {'id': 'a', 'input': {'text': 'alpha'}, 'gold': 'yes'},
            {'id': 'b', 'input': {'text': 'beta'}, 'gold': 'no'},
            {'id': 'c', 'input': {'text': 'gamma'}, 'gold': 'yes'},
        ]
        selector = GenICLSelector(
            pool, FakeGenICLEncoder(), FakeGenICLScorer(), 'toy',
            candidate_pool_size=2, n_prefix_tokens=3,
        )
        selected = selector.select_many([{'text': 'alpha query'}], 1)[0]
        self.assertEqual(selected[0]['id'], 'b')
        self.assertEqual(selected[0]['e5_rank'], 2)
        self.assertEqual(latent_description('toy', 3), '<toy-0><toy-1><toy-2>')

    def test_runner_keeps_failed_samples_in_denominator(self):
        # 正确、无效、异常各一条，检查 JSONL 与汇总指标一致。
        samples = [{'input': {'text': str(i)}, 'prompt': str(i), 'gold': 'yes'} for i in range(3)]
        task = TaskData('toy', ('yes', 'no'), samples)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, 'query_vllm', side_effect=['yes', 'unknown', RuntimeError('offline')]):
                report = runner.evaluate_task(task, 'qwen', 'model', 'http://unused', Path(directory), 1, 0, True)
            rows = [json.loads(line) for line in Path(report['output']).read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(report['correct'], 1)
        self.assertEqual(report['invalid'], 2)
        self.assertEqual(report['request_errors'], 1)
        self.assertAlmostEqual(report['accuracy'], 1 / 3)

    def test_random_selector_and_icl_prompt_are_reproducible(self):
        # 同一 seed 和测试样本必须选择同一组无重复训练示例。
        pool = [
            {'id': f'train_{i}', 'input': {'text': f'example {i}'}, 'gold': 'yes'}
            for i in range(10)
        ]
        first = select_random_examples(pool, 4, 42, 'banking77', 0)
        repeated = select_random_examples(pool, 4, 42, 'banking77', 0)
        other_sample = select_random_examples(pool, 4, 42, 'banking77', 1)
        self.assertEqual(first, repeated)
        self.assertEqual(len({item['id'] for item in first}), 4)
        self.assertNotEqual(first, other_sample)

        base_prompt = prompts.build_banking77_prompt('test request', ('yes', 'no'))
        icl_prompt = prompts.build_icl_prompt('banking77', base_prompt, first)
        self.assertIn('Example 1:', icl_prompt)
        self.assertIn('Intent label: yes', icl_prompt)
        # 待测问题必须位于全部演示之后，防止模型回答了训练示例。
        self.assertGreater(icl_prompt.rfind('test request'), icl_prompt.rfind('Example 4:'))

    def test_random_runner_records_selected_demonstrations(self):
        samples = [{
            'input': {'text': 'test request'},
            'prompt': prompts.build_banking77_prompt('test request', ('yes', 'no')),
            'gold': 'yes',
        }]
        pool = [
            {'id': f'train_{i}', 'input': {'text': f'example {i}'}, 'gold': 'yes'}
            for i in range(5)
        ]
        task = TaskData('banking77', ('yes', 'no'), samples, demonstration_pool=pool)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, 'query_vllm', return_value='yes') as query:
                report = runner.evaluate_task(
                    task, 'qwen', 'model', 'http://unused', Path(directory), 1, 0,
                    True, method='random', num_shots=2, seed=7,
                )
            row = json.loads(Path(report['output']).read_text())
        self.assertEqual(report['method'], 'random')
        self.assertEqual(report['num_shots'], 2)
        self.assertEqual(len(row['demonstrations']), 2)
        self.assertIn('Example 2:', query.call_args.kwargs['prompt'])

    def test_bm25_selector_retrieves_relevant_examples(self):
        # BM25 只能根据输入文本检索，最相关的 card delivery 示例应排在首位。
        pool = [
            {'id': 'cash', 'input': {'text': 'cash withdrawal from an ATM'}, 'gold': 'a'},
            {'id': 'card', 'input': {'text': 'my new card has not arrived'}, 'gold': 'b'},
            {'id': 'transfer', 'input': {'text': 'international bank transfer'}, 'gold': 'c'},
        ]
        selector = BM25Selector(pool)
        selected = selector.select({'text': 'where is my card delivery'}, 2)
        self.assertEqual(selected[0]['id'], 'card')
        self.assertGreater(selected[0]['selection_score'], selected[1]['selection_score'])
        # 选择器返回副本，不能把分数写回公共训练池。
        self.assertNotIn('selection_score', pool[1])

    def test_bm25_runner_records_scores_and_prompt(self):
        samples = [{
            'input': {'text': 'my card delivery is late'},
            'prompt': prompts.build_banking77_prompt(
                'my card delivery is late', ('card_arrival', 'cash_withdrawal')
            ),
            'gold': 'card_arrival',
        }]
        pool = [
            {
                'id': 'train_cash',
                'input': {'text': 'cash withdrawal at an ATM'},
                'gold': 'cash_withdrawal',
            },
            {
                'id': 'train_card',
                'input': {'text': 'card delivery has not arrived'},
                'gold': 'card_arrival',
            },
        ]
        task = TaskData('banking77', ('card_arrival', 'cash_withdrawal'), samples,
                        demonstration_pool=pool)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, 'query_vllm', return_value='card_arrival') as query:
                report = runner.evaluate_task(
                    task, 'qwen', 'model', 'http://unused', Path(directory), 1, 0,
                    True, method='bm25', num_shots=1,
                )
            row = json.loads(Path(report['output']).read_text())
        self.assertEqual(row['demonstrations'][0]['id'], 'train_card')
        self.assertGreater(row['demonstrations'][0]['selection_score'], 0)
        self.assertIn('card delivery has not arrived', query.call_args.kwargs['prompt'])
        self.assertEqual(report['bm25_k1'], 1.5)
        self.assertEqual(report['bm25_b'], 0.75)

    def test_kate_selector_uses_cosine_and_preserves_prompt_order(self):
        pool = [
            {'id': 'cash', 'input': {'text': 'cash withdrawal from an ATM'}, 'gold': 'a'},
            {'id': 'card', 'input': {'text': 'my new card has not arrived'}, 'gold': 'b'},
            {'id': 'transfer', 'input': {'text': 'international bank transfer'}, 'gold': 'c'},
        ]
        selector = KATESelector(pool, FakeDenseEncoder(), 'banking77')
        selected = selector.select_many(
            [{'text': 'where is my card delivery'}], 2, 'most_similar_first'
        )[0]
        self.assertEqual([item['id'] for item in selected], ['card', 'transfer'])
        self.assertGreater(selected[0]['selection_score'], selected[1]['selection_score'])
        self.assertEqual([item['selection_rank'] for item in selected], [1, 2])
        self.assertNotIn('selection_score', pool[1])

        reversed_prompt = selector.select_many(
            [{'text': 'where is my card delivery'}], 2, 'most_similar_last'
        )[0]
        self.assertEqual([item['id'] for item in reversed_prompt], ['transfer', 'card'])
        self.assertEqual([item['selection_rank'] for item in reversed_prompt], [2, 1])
        self.assertEqual([item['prompt_position'] for item in reversed_prompt], [1, 2])

    def test_kate_embedding_cache_avoids_reencoding_training_pool(self):
        pool = [
            {'id': 'cash', 'input': {'text': 'cash withdrawal from an ATM'}, 'gold': 'a'},
            {'id': 'card', 'input': {'text': 'my new card has not arrived'}, 'gold': 'b'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            first_encoder = FakeDenseEncoder()
            KATESelector(pool, first_encoder, 'banking77', Path(directory))

            second_encoder = FakeDenseEncoder()
            second_encoder.vectors.pop('cash withdrawal from an ATM')
            second_encoder.vectors.pop('my new card has not arrived')
            # 如果缓存没有命中，此处会因缺少训练文本向量而失败。
            cached = KATESelector(pool, second_encoder, 'banking77', Path(directory))
            self.assertEqual(cached.corpus_embeddings.shape, (2, 3))

    def test_kate_runner_uses_precomputed_shared_selections(self):
        sample_prompt = prompts.build_banking77_prompt(
            'where is my card', ('card_arrival', 'cash_withdrawal')
        )
        demonstration = {
            'id': 'train_card',
            'input': {'text': 'my card has not arrived'},
            'gold': 'card_arrival',
            'selection_score': 0.93,
            'selection_rank': 1,
            'prompt_position': 1,
        }
        task = TaskData(
            'banking77',
            ('card_arrival', 'cash_withdrawal'),
            [{'input': {'text': 'where is my card'}, 'prompt': sample_prompt,
              'gold': 'card_arrival'}],
            demonstration_pool=[demonstration],
            precomputed_selections=[[demonstration]],
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, 'query_vllm', return_value='card_arrival') as query:
                report = runner.evaluate_task(
                    task, 'qwen', 'model', 'http://unused', Path(directory), 1, 0,
                    True, method='kate', num_shots=1,
                    kate_encoder='fake', kate_order='most_similar_first',
                    kate_max_length=128,
                )
            row = json.loads(Path(report['output']).read_text())
        self.assertEqual(row['demonstrations'][0]['id'], 'train_card')
        self.assertIn('my card has not arrived', query.call_args.kwargs['prompt'])
        self.assertEqual(report['kate_encoder'], 'fake')
        self.assertEqual(report['kate_order'], 'most_similar_first')

    def test_ceil_selector_uses_conditional_dpp_and_runner_contract(self):
        class FakeCEILEncoder:
            temperature = 1.0
            candidate_pool_size = 3
            vectors = {
                'query': [1.0, 0.0],
                'nearly duplicate one': [1.0, 0.0],
                'nearly duplicate two': [0.999, 0.045],
                'relevant but diverse': [0.8, 0.6],
            }

            def encode(self, texts):
                import numpy as np
                return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

        pool = [
            {'id': 'a', 'input': {'text': 'nearly duplicate one'}, 'gold': 'x'},
            {'id': 'b', 'input': {'text': 'nearly duplicate two'}, 'gold': 'x'},
            {'id': 'c', 'input': {'text': 'relevant but diverse'}, 'gold': 'y'},
        ]
        selector = CEILSelector(pool, FakeCEILEncoder())
        selected = selector.select_many([{'text': 'query'}], 2)[0]
        self.assertEqual([item['id'] for item in selected], ['a', 'c'])
        self.assertEqual([item['prompt_position'] for item in selected], [1, 2])
        self.assertEqual(len(greedy_dpp_map(selector.corpus_embeddings @ selector.corpus_embeddings.T + 1e-5 * __import__('numpy').eye(3), 2)), 2)

        sample_prompt = prompts.build_banking77_prompt('query', ('x', 'y'))
        task = TaskData(
            'banking77', ('x', 'y'),
            [{'input': {'text': 'query'}, 'prompt': sample_prompt, 'gold': 'x'}],
            demonstration_pool=pool,
            precomputed_selections=[selected],
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, 'query_vllm', return_value='x'):
                report = runner.evaluate_task(
                    task, 'qwen', 'model', 'http://unused', Path(directory), 1, 0,
                    True, method='ceil', num_shots=2, ceil_checkpoint='fake-checkpoint',
                )
        self.assertEqual(report['ceil_checkpoint'], 'fake-checkpoint')


if __name__ == '__main__':
    unittest.main()
