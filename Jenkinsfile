pipeline {
  agent { label 'myll_ia_bakastov' }

  options {
    timestamps()
    timeout(time: 12, unit: 'MINUTES')
    disableConcurrentBuilds()
  }

  environment {
    PORTS = "8092 8093 8094 8095"
  }

  stages {

    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Build wheel/sdist') {
      steps {
        sh '''#!/usr/bin/env bash
          set -eux
          python3 -V
          rm -rf .venv .venv_test dist build *.egg-info run_output app-restoringvalues.tgz

          python3 -m venv .venv
          . .venv/bin/activate

          python -m pip install -U pip
          python -m pip install build
          python -m build

          ls -la dist
        '''
      }
    }

    stage('Install wheel into clean venv') {
      steps {
        sh '''#!/usr/bin/env bash
          set -eux
          python3 -m venv .venv_test
          . .venv_test/bin/activate

          python -m pip install -U pip
          python -m pip install dist/*.whl

          python -c "import restoringvalues; print(restoringvalues.__version__)"
          restoringvalues-run --help >/dev/null
        '''
      }
    }

    stage('Smoke run (90s, no GUI)') {
      steps {
        sh '''#!/usr/bin/env bash
set -euo pipefail
. .venv_test/bin/activate

mkdir -p run_output

echo "==> AGENT USER: $(whoami)"
id || true

echo "==> Cleanup old outputs"
rm -f Reciever/*.csv Business/*.csv Business/data_out_*.csv Business/data_metrics_*.csv run_output/*.pid run_output/*.log || true

echo "==> Listeners before cleanup"
ss -lntp | egrep ':8092|:8093|:8094|:8095' || true

kill_by_ports() {
  for p in $PORTS; do
    fuser -kv ${p}/tcp >/dev/null 2>&1 || true
  done
}

ports_free_once() {
  for p in $PORTS; do
    if ss -lnt | grep -q ":${p} "; then
      return 1
    fi
  done
  return 0
}

ports_free_stable_or_fail() {
  for t in 1 2 3; do
    if ! ports_free_once; then
      return 1
    fi
    sleep 0.5
  done
  return 0
}

echo "==> Cleanup ports"
kill_by_ports
sleep 1

echo "==> Verify ports are FREE"
if ! ports_free_stable_or_fail; then
  echo "Ports not free after cleanup"
  ss -lntp | egrep ':8092|:8093|:8094|:8095' || true
  exit 1
fi

start_bg() {
  local name="$1"; shift
  setsid nohup "$@" > "run_output/${name}.log" 2>&1 & echo $! > "run_output/${name}.pid"
  echo "Started $name pid=$(cat run_output/${name}.pid)"
}

stop_group() {
  local name="$1"
  if [ -f "run_output/${name}.pid" ]; then
    pid="$(cat run_output/${name}.pid)"
    echo "Stopping $name group (pid=$pid)"
    kill -- "-$pid" >/dev/null 2>&1 || true
  fi
}

wait_ports() {
  for i in $(seq 1 60); do
    ok=0
    for p in $PORTS; do
      if ss -lnt | grep -q ":${p} "; then ok=$((ok+1)); fi
    done
    if [ "$ok" -eq 4 ]; then return 0; fi
    sleep 0.5
  done
  return 1
}

fail_with_logs() {
  echo "==> FAILURE diagnostics"
  ss -lntp | egrep ':8092|:8093|:8094|:8095' || true
  tail -n 200 run_output/*.log 2>/dev/null || true
}

echo "==> Start Reciever FIRST"
start_bg reciever python3 Reciever/reciever.py

echo "==> Wait ports 8092-8095"
if ! wait_ports; then
  echo "Ports not ready after Reciever start"
  fail_with_logs
  stop_group reciever
  kill_by_ports
  exit 1
fi

echo "==> Start Simulator"
start_bg simulator python3 Simulator/simulator.py

echo "==> Wait CSV from Reciever"
for i in $(seq 1 60); do
  cnt=$(ls Reciever/data_port_*.csv 2>/dev/null | wc -l || true)
  if [ "$cnt" -ge 4 ]; then break; fi
  sleep 0.5
done

echo "==> Start Business"
start_bg business python3 Business/business.py

echo "==> Let them work 90s"
sleep 90

echo "==> Copy outputs"
cp -a Reciever/*.csv run_output/ 2>/dev/null || true
find Business -maxdepth 3 -type f -name '*.csv' -exec cp -a {} run_output/ \\; 2>/dev/null || true

echo "==> Stop processes"
stop_group business
stop_group simulator
stop_group reciever

echo "==> Final port cleanup"
kill_by_ports
sleep 1

echo "Smoke run OK"
'''
      }
    }

    stage('Package app artifact (tgz)') {
      steps {
        sh '''#!/usr/bin/env bash
          set -eux
          tar -czf app-restoringvalues.tgz \
            Simulator Reciever Business GUI \
            requirements.txt \
            pyproject.toml setup.cfg setup.py 2>/dev/null || true

          ls -la app-restoringvalues.tgz
        '''
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'dist/*, run_output/*, app-restoringvalues.tgz', fingerprint: true, allowEmptyArchive: true
      cleanWs()
    }
  }
}