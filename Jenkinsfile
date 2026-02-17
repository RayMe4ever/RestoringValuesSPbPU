pipeline {
  agent { label 'myll_ia_bakastov' }
  options {
    timestamps()
    timeout(time: 12, unit: 'MINUTES')
    disableConcurrentBuilds()
  }

  environment {
    VENV  = ".venv"
    PORTS = "8092 8093 8094 8095"
  }

  stages {

    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Prepare venv + deps') {
      steps {
        sh '''#!/usr/bin/env bash
          set -eux
          python3 -V
          rm -rf "$VENV" run_output artifacts.tgz || true
          python3 -m venv "$VENV"
          . "$VENV/bin/activate"
          python -m pip install -U pip
          python -m pip install -r requirements.txt
        '''
      }
    }

    stage('Smoke run (robust, 90s)') {
      steps {
        sh '''#!/usr/bin/env bash
set -euo pipefail
. "$VENV/bin/activate"

mkdir -p run_output

echo "==> Cleanup old outputs"
rm -f Reciever/*.csv Business/*.csv Business/data_out_*.csv Business/data_metrics_*.csv run_output/*.pid run_output/*.log || true

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
    ports_free_once || return 1
    sleep 0.5
  done
  return 0
}

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

show_diag() {
  echo "==> DIAGNOSTICS"
  echo "-- listeners:"
  ss -lntp | egrep ':8092|:8093|:8094|:8095' || true
  echo "-- reciever.log (tail):"
  tail -n 120 run_output/reciever.log 2>/dev/null || true
  echo "-- simulator.log (tail):"
  tail -n 120 run_output/simulator.log 2>/dev/null || true
  echo "-- business.log (tail):"
  tail -n 120 run_output/business.log 2>/dev/null || true
  echo "-- files:"
  ls -la Reciever 2>/dev/null || true
  ls -la Business 2>/dev/null || true
  find Reciever -maxdepth 1 -type f -name '*.csv' -print 2>/dev/null || true
  find Business -maxdepth 3 -type f -name '*.csv' -print 2>/dev/null || true
}

wait_ready() {
  for i in $(seq 1 80); do
    if ss -lnt | egrep -q ':8092 |:8093 |:8094 |:8095 '; then
      return 0
    fi
    if ls Reciever/*.csv >/dev/null 2>&1 || find Business -maxdepth 3 -type f -name '*.csv' >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

echo "==> Pre-clean ports"
kill_by_ports
sleep 1
if ! ports_free_stable_or_fail; then
  echo "Ports are busy before start."
  show_diag
  exit 1
fi

echo "==> Start components (background)"
start_bg reciever  python3 Reciever/reciever.py
start_bg simulator python3 Simulator/simulator.py
start_bg business  python3 Business/business.py

echo "==> Wait readiness (ports or CSVs)"
if ! wait_ready; then
  echo "Services did not become ready in time."
  show_diag
  stop_group business
  stop_group simulator
  stop_group reciever
  kill_by_ports
  exit 1
fi

echo "==> Let them work 90s"
sleep 90

echo "==> Collect outputs"
cp -a Reciever/*.csv run_output/ 2>/dev/null || true
find Business -maxdepth 3 -type f -name '*.csv' -exec cp -a {} run_output/ \\; 2>/dev/null || true

echo "==> Post-check: require some output"
REC_COUNT="$(ls Reciever/*.csv 2>/dev/null | wc -l | tr -d ' ')"
BUS_COUNT="$(find Business -maxdepth 3 -type f -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')"
echo "Reciever CSV count: ${REC_COUNT}"
echo "Business CSV count: ${BUS_COUNT}"

if [ "${REC_COUNT}" -eq 0 ] && [ "${BUS_COUNT}" -eq 0 ]; then
  echo "No CSV produced by either Reciever or Business."
  show_diag
  stop_group business
  stop_group simulator
  stop_group reciever
  kill_by_ports
  exit 1
fi

echo "==> Stop processes"
stop_group business
stop_group simulator
stop_group reciever

echo "==> Final port cleanup"
kill_by_ports
sleep 1

echo "==> Verify ports are free after stopping"
if ! ports_free_stable_or_fail; then
  echo "Ports still busy after stopping."
  show_diag
  exit 1
fi

echo "Smoke run OK"
'''
      }
    }

    stage('Package artifacts (tgz)') {
      steps {
        sh '''#!/usr/bin/env bash
          set -eux
          tar -czf artifacts.tgz run_output Simulator Reciever Business GUI requirements.txt 2>/dev/null || true
          ls -la artifacts.tgz
        '''
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'run_output/*, Reciever/*.csv, Business/*.csv, artifacts.tgz', fingerprint: true, allowEmptyArchive: true
      cleanWs()
    }
  }
}