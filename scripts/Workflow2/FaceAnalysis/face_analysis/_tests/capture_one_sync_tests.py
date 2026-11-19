import json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import face_analysis.run_face_analysis as app

class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input'
        self.out = self.root / 'out'
        self.source.mkdir()
        (self.out / '_Embeddings').mkdir(parents=True)
        self.name = 'Школа_270016.jpg'
        self.ph = 'PH_' + self.name
        self.save('info_faces.json', {self.name: {'filename': self.name, 'faces': [{'face_index': 0, 'bbox': [1], 'person': 'Анна'}]}})
        self.save('info_faces_landmarks.json', {self.name: {'filename': self.name, 'faces': [{'landmark_2d_106': [1]}]}})
        self.save('sync_state.json', {self.name: {'mtime': 1, 'size': 1}})
        self.save('_Embeddings/faces_index.json', {self.name: [0]})
        np.save(self.out / '_Embeddings/faces_embeddings.npy', np.ones((1, 512), dtype=np.float32))
        self.cfg = SimpleNamespace(a_af_config_file='x', a_af_model_name='test', a_af_det_size=640, a_af_det_thresh=None, a_af_tasks=[], a_af_output_dir=str(self.out), a_af_input_dir=str(self.source), a_af_mode='sync', all_threads=1)
        self.calls = []
    def save(self, name, value):
        (self.out / name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    def read(self, name):
        return json.loads((self.out / name).read_text(encoding='utf-8'))
    def run_app(self, face=True):
        def process(analyzer, path):
            self.calls.append(path.name)
            return path.name, (([{'bbox': [5], 'landmark_2d_106': [9]}], [np.ones(512, dtype=np.float32)], (200, 200)) if face else (None, None, (200, 200)))
        cfg = SimpleNamespace(config={'model': {}, 'task_flags': {}})
        with patch.object(app, 'get_config', return_value=self.cfg), patch.object(app, 'ConfigManager', return_value=cfg), patch.object(app, 'FaceAnalyzer', return_value=MagicMock()), patch.object(app, 'load_and_process_task', side_effect=process), patch.object(app, 'tqdm', side_effect=lambda items, **kw: items), patch.object(app.pysm_context, 'log_html'), patch.object(app, 'StandardTreeBuilder'):
            try:
                app.main()
            except SystemExit as exc:
                if exc.code != 0:
                    raise
    def test_export_and_repeated_run(self):
        (self.source / self.name).write_bytes(b'original')
        (self.source / self.ph).write_bytes(b'export')
        self.run_app()
        data = self.read('info_faces.json')
        self.assertEqual(set(data), {self.name})
        self.assertEqual(data[self.name]['faces'][0]['bbox'], [5])
        self.assertEqual(data[self.name]['faces'][0]['person'], 'Анна')
        self.assertEqual(self.read('info_faces_landmarks.json')[self.name]['faces'][0]['landmark_2d_106'], [9])
        self.assertEqual(self.read('_Embeddings/faces_index.json'), {self.name: [0]})
        self.assertEqual(self.read('sync_state.json')[self.ph]['sync_target'], self.name)
        self.run_app()
        self.assertEqual(self.calls, [self.ph])
    def test_failed_update_retries(self):
        (self.source / self.ph).write_bytes(b'export')
        self.run_app(False)
        self.assertNotIn(self.ph, self.read('sync_state.json'))
        self.run_app()
        self.assertEqual(self.calls, [self.ph, self.ph])
    def test_unknown_export_is_new(self):
        (self.source / 'PH_new.jpg').write_bytes(b'new')
        self.run_app()
        self.assertIn('PH_new.jpg', self.read('info_faces.json'))
    def test_legacy_state_reprocessed(self):
        p = self.source / self.ph
        p.write_bytes(b'export')
        st = p.stat()
        self.save('sync_state.json', {self.ph: {'mtime': st.st_mtime, 'size': st.st_size}})
        self.run_app()
        self.assertEqual(self.calls, [self.ph])
        self.assertEqual(self.read('sync_state.json')[self.ph]['sync_target'], self.name)
    def test_switching_between_export_and_original_reprocesses_each(self):
        original = self.source / self.name
        exported = self.source / self.ph
        original.write_bytes(b'original')
        exported.write_bytes(b'export')
        self.run_app()
        saved_export = self.root / self.ph
        exported.replace(saved_export)
        self.run_app()
        saved_export.replace(exported)
        self.run_app()
        self.assertEqual(self.calls, [self.ph, self.name, self.ph])
        self.assertEqual(set(self.read('info_faces.json')), {self.name})

    def test_failed_main_json_write_is_retried(self):
        (self.source / self.ph).write_bytes(b'export')
        save = app.save_json_atomic
        def fail_main(path, data, indent=2):
            if path.name == 'info_faces.json':
                raise OSError('write failed')
            return save(path, data, indent)
        with patch.object(app, 'save_json_atomic', side_effect=fail_main):
            self.run_app()
        self.assertNotIn(self.ph, self.read('sync_state.json'))
        self.run_app()
        self.assertEqual(self.calls, [self.ph, self.ph])

    def test_missing_original_state_still_reprocesses_original_after_export(self):
        self.save('sync_state.json', {})
        original = self.source / self.name
        exported = self.source / self.ph
        original.write_bytes(b'original')
        exported.write_bytes(b'export')
        self.run_app()
        exported.unlink()
        self.run_app()
        self.assertEqual(self.calls, [self.ph, self.name])
        self.assertEqual(self.read('info_faces.json')[self.name]['faces'][0]['person'], 'Анна')

    def test_create_unchanged(self):
        self.cfg.a_af_mode = 'create'
        (self.source / self.ph).write_bytes(b'export')
        self.run_app()
        self.assertEqual(set(self.read('info_faces.json')), {self.ph})

if __name__ == '__main__':
    unittest.main()
