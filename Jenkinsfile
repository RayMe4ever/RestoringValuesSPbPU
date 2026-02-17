pipeline {
  agent { label 'myll_ia_bakastov' }

  options {
    timestamps()
    disableConcurrentBuilds()
  }

  environment {
    VENV = ".venv"
    HOST = "127.0.0.1"
  }

  stages {

    stage('Checkout') {
      steps {
        checkout scm
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
          for p in 8050 8051 8092 8093 8094 8095; do
            fuser -k ${p}/tcp 2>/dev/null || true
          done
          set -e
        '''
      }
    }

    stage('Run services') {
      steps {
        sh '''
          set -e
          . "$VENV/bin/activate"

          export WEBSOCKET_HOST=$HOST

          mkdir -p run_output

          echo "==> Start Simulator (WS server + sender)"
          nohup python Simulator/simulator.py \
            > run_output/simulator.log 2>&1 &

          sleep 5

          echo "==> Start Reciever"
          nohup python Reciever/reciever.py \
            > run_output/reciever.log 2>&1 &

          sleep 5

          echo "==> Start Business"
          nohup python Business/business.py \
            > run_output/business.log 2>&1 &

          sleep 5

          echo "==> Start Dash GUI (8050)"
          nohup python GUI/dash_app_test.py \
            > run_output/gui.log 2>&1 &

          echo "Services started"

          # Держим билд живым
          tail -f run_output/gui.log
        '''
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'run_output/*.log', allowEmptyArchive: true
    }
  }
}