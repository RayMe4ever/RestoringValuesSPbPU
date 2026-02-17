pipeline {
  agent { label 'myll_ia_bakastov' }

  options {
    timestamps()
    disableConcurrentBuilds()
  }

  environment {
    VENV = ".venv"
    HOST = "127.0.0.1"
    PYTHONUNBUFFERED = "1"
  }

  stages {

    stage('Checkout') {
      steps {
        checkout scm
        sh '''
          set -e
          echo "==> Git info"
          git rev-parse --abbrev-ref HEAD || true
          git rev-parse HEAD || true
          git log -1 --oneline || true
        '''
      }
    }

    stage('Prepare venv') {
      steps {
        sh '''
          set -e
          python3 -m venv "$VENV"
          . "$VENV/bin/activate"
          pip install -U pip
          pip install -r requirements.txt
        '''
      }
    }

    stage('Clean ports') {
      steps {
        sh '''
          set +e
          # Важно: добавили 8000 (HTTP API Business) — иначе прошлый job может держать порт и ломать новый запуск.
          for p in 8000 8050 8051 8092 8093 8094 8095; do
            fuser -k ${p}/tcp 2>/dev/null || true
          done
          set -e
        '''
      }
    }

    stage('Run services') {
      steps {
        sh '''
          set -euo pipefail
          . "$VENV/bin/activate"

          export WEBSOCKET_HOST=$HOST
          export PYTHONUNBUFFERED=$PYTHONUNBUFFERED

          mkdir -p run_output
          rm -f run_output/*.pid

          cleanup() {
            set +e
            echo "==> Cleanup: stopping services"
            # Штатно гасим PID-ы
            for f in run_output/*.pid; do
              [ -f "$f" ] || continue
              pid=$(cat "$f" 2>/dev/null || true)
              if [ -n "${pid}" ]; then
                kill "${pid}" 2>/dev/null || true
              fi
            done

            # Дадим процессам шанс завершиться
            sleep 2

            # Жёсткое добивание, если ещё живы
            for f in run_output/*.pid; do
              [ -f "$f" ] || continue
              pid=$(cat "$f" 2>/dev/null || true)
              if [ -n "${pid}" ]; then
                kill -9 "${pid}" 2>/dev/null || true
              fi
            done

            # На всякий случай — чистим порты
            for p in 8000 8050 8051 8092 8093 8094 8095; do
              fuser -k ${p}/tcp 2>/dev/null || true
            done
            echo "==> Cleanup done"
          }

          trap cleanup EXIT INT TERM

          start_bg() {
            name="$1"; shift
            log="run_output/${name}.log"
            pidfile="run_output/${name}.pid"

            echo "==> Start ${name}"
            nohup "$@" > "${log}" 2>&1 &
            echo $! > "${pidfile}"
            echo "    pid=$(cat "${pidfile}")  log=${log}"
          }

          start_bg simulator python Simulator/simulator.py
          sleep 5

          start_bg reciever python Reciever/reciever.py
          sleep 5

          start_bg business python Business/business.py
          sleep 5

          start_bg gui python GUI/dash_app_test.py

          echo "==> Services started"
          echo "==> Tail logs (CTRL+C / abort build to stop)"
          tail -n +1 -F run_output/simulator.log run_output/reciever.log run_output/business.log run_output/gui.log
        '''
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'run_output/*.log,run_output/*.pid', allowEmptyArchive: true
    }
  }
}
